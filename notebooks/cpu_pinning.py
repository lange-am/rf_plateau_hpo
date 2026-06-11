"""
cpu_pinning.py
--------------

This module provides a lightweight process scheduler for running many independent
calls of a user-supplied function (e.g., `run_experiment`) in parallel while pinning
each worker process to a disjoint set of physical CPU cores.

The goal is to reduce run-to-run interference in wall-clock timings by preventing
concurrent runs from sharing physical cores (and, when SMT/Hyper-Threading is enabled,
keeping sibling logical CPUs within the same run).

Key Features
------------
- CPU affinity pinning per worker process (Linux).
- Optional SMT usage: allocate sibling logical CPUs for each pinned physical core.
- Queue-based scheduling: a fixed number of pinned workers consumes param_list tasks.
- Optional injection of `n_jobs` based on the allocated physical cores and SMT setting.
- Conflict-safe parameter merging via `merge_safe` to enforce a strict separation between
  static_params and per-run params.

Notes
-----
- This module does not provide full isolation of memory bandwidth / LLC / NUMA effects.
  It only enforces CPU-core exclusivity for concurrent runs.
- For best results, prefer running on an otherwise idle node and use start_method="fork"
  on Linux, especially when static_params include large arrays.

Copyright (c) 2025-2026 Andrey Lange and rf_plateau_hpo contributors.
Licensed under the MIT License. See the LICENSE file in the project root for details.

"""
import os
import sys
import inspect
import traceback
import subprocess
from pathlib import Path
from collections import defaultdict
from multiprocessing import get_context
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from merge_safe import merge_safe

try:
    from tqdm.auto import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    tqdm = None

def _set_thread_env_limits() -> None:
    # Avoid oversubscription from BLAS/OpenMP thread pools inside each process.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _get_allowed_cpus() -> List[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 1
        return list(range(n))


def _cpu_topology(allowed_cpus: Sequence[int]) -> Dict[Tuple[int, int], List[int]]:
    """Return mapping (socket_id, core_id) -> sorted logical CPU IDs, restricted to allowed_cpus."""
    allowed = set(allowed_cpus)
    groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    try:
        out = subprocess.check_output(
            ["lscpu", "--parse=CPU,CORE,SOCKET"],
            text=True,
        )
        for line in out.splitlines():
            if not line or line.startswith("#"):
                continue
            cpu_s, core_s, sock_s = line.split(",")[:3]
            cpu = int(cpu_s)
            if cpu not in allowed:
                continue
            core = int(core_s)
            sock = int(sock_s)
            groups[(sock, core)].append(cpu)

    except Exception:
        base = Path("/sys/devices/system/cpu")
        for cpu in allowed_cpus:
            topo = base / f"cpu{cpu}" / "topology"
            if not topo.exists():
                continue
            sock = int((topo / "physical_package_id").read_text().strip())
            core = int((topo / "core_id").read_text().strip())
            groups[(sock, core)].append(cpu)

    return {k: sorted(v) for k, v in groups.items()}


def _pin_process(cpu_list: Sequence[int]) -> None:
    os.sched_setaffinity(0, set(cpu_list))


def _try_pin_process(cpu_list, *, pinning_policy: str, context: str = "") -> None:
    policy = pinning_policy.lower().strip()
    if policy not in {"ok", "warn", "error"}:
        raise ValueError("pinning_policy must be one of: 'ok', 'warn', 'error'.")

    try:
        _pin_process(cpu_list)
    except Exception as e:
        msg = f"[{context}] Warning: could not set CPU affinity ({type(e).__name__}: {e})"
        if policy == "error":
            raise
        if policy == "warn":
            print(msg, file=sys.stderr, flush=True)
        # policy == "ok": ignore silently


def _func_accepts_n_jobs(func: Callable[..., Any]) -> bool:
    sig = inspect.signature(func)
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return "n_jobs" in sig.parameters


def _allocate_run_cpu_blocks(
    allowed_cpus: Sequence[int],
    n_phys_cores_per_run: int,
    use_smt: Union[bool, str],
    socket_policy: str,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    topo = _cpu_topology(allowed_cpus)
    if not topo:
        raise RuntimeError("CPU topology is empty; cannot allocate run CPU blocks.")

    threads_per_core_visible = max(len(v) for v in topo.values())
    use_smt_resolved = (threads_per_core_visible > 1) if use_smt == "auto" else bool(use_smt)

    socket_policy = socket_policy.lower().strip()
    if socket_policy not in {"prefer", "strict", "none"}:
        raise ValueError("socket_policy must be one of: 'prefer', 'strict', 'none'.")

    if n_phys_cores_per_run <= 0:
        raise ValueError("n_phys_cores_per_run must be positive.")

    by_socket: Dict[int, List[List[int]]] = defaultdict(list)
    for (sock, core) in sorted(topo.keys()):
        sibs = topo[(sock, core)]
        by_socket[sock].append(sibs if use_smt_resolved else [sibs[0]])

    blocks: List[List[int]] = []
    leftovers: List[List[int]] = []

    if socket_policy == "none":
        all_cores: List[List[int]] = []
        for sock in sorted(by_socket):
            all_cores.extend(by_socket[sock])
        full = (len(all_cores) // n_phys_cores_per_run) * n_phys_cores_per_run
        for i in range(0, full, n_phys_cores_per_run):
            chunk = all_cores[i : i + n_phys_cores_per_run]
            blocks.append([cpu for core_list in chunk for cpu in core_list])

    else:
        for sock in sorted(by_socket):
            cores = by_socket[sock]
            full = (len(cores) // n_phys_cores_per_run) * n_phys_cores_per_run
            for i in range(0, full, n_phys_cores_per_run):
                chunk = cores[i : i + n_phys_cores_per_run]
                blocks.append([cpu for core_list in chunk for cpu in core_list])
            leftovers.extend(cores[full:])

        if socket_policy == "prefer":
            full_left = (len(leftovers) // n_phys_cores_per_run) * n_phys_cores_per_run
            for i in range(0, full_left, n_phys_cores_per_run):
                chunk = leftovers[i : i + n_phys_cores_per_run]
                blocks.append([cpu for core_list in chunk for cpu in core_list])

    n_jobs_value = n_phys_cores_per_run * (threads_per_core_visible if use_smt_resolved else 1)

    info = {
        "threads_per_core_visible": threads_per_core_visible,
        "use_smt_resolved": use_smt_resolved,
        "socket_policy": socket_policy,
        "n_phys_cores_per_run": n_phys_cores_per_run,
        "n_jobs_value": n_jobs_value,
        "n_run_slots": len(blocks),
        "allowed_cpu_count": len(allowed_cpus),
        "sockets_visible": sorted(by_socket.keys()),
    }
    return [sorted(b) for b in blocks], info


def _worker_loop(
    func: Callable[..., Any],
    cpu_list: List[int],
    static_params: Optional[Dict[str, Any]],
    send_outputs: bool,
    inject_n_jobs: bool,
    n_jobs_value: int,
    override_n_jobs: bool,
    set_thread_env_limits: bool,
    pinning_policy: str,
    task_q,
    out_q,
) -> None:
    _try_pin_process(cpu_list, pinning_policy=pinning_policy, context="_worker_loop")

    if set_thread_env_limits:
        _set_thread_env_limits()

    base_kwargs = dict(static_params) if static_params is not None else {}

    while True:
        item = task_q.get()
        if item is None:
            break

        idx, kwargs = item
        try:
            call_kwargs = dict(base_kwargs)
            call_kwargs = merge_safe(call_kwargs, kwargs, context='_worker_loop')

            if inject_n_jobs:
                if override_n_jobs or ("n_jobs" not in call_kwargs):
                    call_kwargs["n_jobs"] = n_jobs_value

            res = func(**call_kwargs)
            out_q.put((idx, res if send_outputs else None, None))
        except Exception:
            out_q.put((idx, None, traceback.format_exc()))


def run_queue_pinned(
    func: Callable[..., Any],
    param_list: Sequence[Dict[str, Any]],
    *,
    n_phys_cores_per_run: int = 1,
    allowed_cpus: Optional[Sequence[int]] = None,
    static_params: Optional[Dict[str, Any]] = None,
    use_smt: Union[bool, str] = "auto",
    socket_policy: str = "prefer",
    return_outputs: bool = True,
    max_parallel_runs: Optional[int] = None,
    start_method: str = "fork",
    set_thread_env_limits: bool = True,
    override_n_jobs: bool = True,
    raise_on_error: bool = True,
    pinning_policy: str = "warn",
    verbose: int = 0,
    progress_bar: bool = False,
    progress_bar_position: Optional[int] = None,
) -> Optional[List[Any]]:
    """
    Execute many independent calls of ``func(**params)`` in parallel with per-run CPU affinity pinning.

    This scheduler starts a fixed number of worker processes. Each worker is pinned to a disjoint set
    of physical CPU cores and consumes tasks from a queue. Pinning prevents concurrently running tasks
    from sharing physical cores. If SMT (Hyper-Threading) is used, sibling logical CPUs of a core are
    kept within the same run slot.

    Parameters
    ----------
    func:
        Callable invoked as ``func(**call_kwargs)`` for each task.
        For ``start_method="spawn"``, this callable must be picklable (typically defined at module scope).
    param_list:
        Sequence of per-run parameter dictionaries. Each element contains run-specific arguments only
        (e.g., method, delta, n_trials, random_state, outdir). By default, keys must not overlap with
        ``static_params``; conflicts raise an error during merging.
    n_phys_cores_per_run:
        Number of physical CPU cores allocated to each concurrent run slot (each worker).
    allowed_cpus:
        Optional explicit list of logical CPU IDs from which run slots are built.
        If None, use the current process affinity mask returned by
        os.sched_getaffinity(0). The provided list must be a subset of the current
        allowed CPU set; this prevents bypassing Slurm/cpuset restrictions.
    static_params:
        Shared parameters applied to every run (e.g., X, y, problem, score_func, greater_is_better,
        class_weight). These parameters are stored once per worker and are not sent through the task queue
        for each run. If ``None``, no shared parameters are applied.
    use_smt:
        ``True`` / ``False`` / ``"auto"``. If True, each pinned physical core contributes all visible
        sibling logical CPUs (typically 2). If False, each pinned physical core contributes exactly one
        logical CPU. If ``"auto"``, SMT is considered enabled when more than one sibling is visible in the
        current allowed CPU set.
    socket_policy:
        CPU allocation policy with respect to sockets/NUMA nodes:
          - ``"prefer"``: allocate as many single-socket run slots as possible, then fill from leftovers
            (some slots may mix sockets).
          - ``"strict"``: allocate only single-socket run slots; leftover cores are ignored.
          - ``"none"``: ignore sockets and allocate sequentially across all available cores.
    return_outputs:
        If True, return a list of results aligned with ``param_list`` order. If False, return None.
    max_parallel_runs:
        Optional upper bound on the number of concurrent run slots (workers). If None, uses as many run slots
        as can be allocated from the available CPUs.
    start_method:
        Multiprocessing start method (e.g., ``"fork"``, ``"spawn"``, ``"forkserver"``).
        On Linux, ``"fork"`` is typically fastest and most memory-efficient when ``static_params`` includes
        large arrays (copy-on-write). ``"spawn"`` provides a clean interpreter but serializes ``static_params``
        once per worker.
    set_thread_env_limits:
        If True, sets ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``, ``NUMEXPR_NUM_THREADS``
        to 1 inside each worker to reduce nested parallelism oversubscription.
    override_n_jobs:
        If True and ``func`` accepts ``n_jobs`` (or has ``**kwargs``), inject ``n_jobs`` based on the allocated
        CPU set:
          - if SMT is used: ``n_jobs = n_phys_cores_per_run * threads_per_core_visible``
          - otherwise:      ``n_jobs = n_phys_cores_per_run``
        If False, do not overwrite an existing ``n_jobs`` provided in per-run parameters.
    raise_on_error:
        If True, raise a RuntimeError if any task fails. If False, failed tasks yield ``None`` in the results list.
    pinning_policy:
        Behavior when CPU affinity pinning fails in a worker:
          - ``"ok"``: ignore silently,
          - ``"warn"``: print a warning to stderr,
          - ``"error"``: raise and terminate the worker.
    verbose:
        Verbosity level for the scheduler itself (not necessarily the underlying experiment code). ``0`` disables
        scheduler status prints; higher values may print allocation summaries.
    progress_bar : bool, default=False
        If True and tqdm is installed, display a progress bar tracking task completion.
        If tqdm is not installed, a warning is printed and no bar is shown.


    Returns
    -------
    Optional[List[Any]]:
        If ``return_outputs=True``, a list of results in the same order as ``param_list``; otherwise ``None``.

    Guarantees / Limitations
    ------------------------
    Guarantees:
      - Concurrent runs do not share physical CPU cores (via CPU affinity).
      - When SMT is enabled, sibling logical CPUs of a core remain within the same run slot.

    Limitations:
      - Does not guarantee full isolation of memory bandwidth, LLC/L3 cache contention, disk I/O contention,
        or NUMA memory placement. CPU pinning reduces interference but does not eliminate all shared-resource effects.
    """
    if os.name != "posix":
        start_method = "spawn"

    if set_thread_env_limits:
        _set_thread_env_limits()

    static_params_local = dict(static_params) if static_params is not None else None

    allowed_cpu_list = _get_allowed_cpus() if allowed_cpus is None else sorted(set(map(int, allowed_cpus)))

    # Respect externally imposed affinity/cpuset restrictions.
    externally_allowed = set(_get_allowed_cpus())
    not_allowed = set(allowed_cpu_list) - externally_allowed
    if not_allowed:
        raise ValueError(
            f"allowed_cpus contains CPUs outside the current process affinity/cpuset: "
            f"{sorted(not_allowed)}. Current allowed CPUs: {sorted(externally_allowed)}."
        )

    run_cpu_blocks, info = _allocate_run_cpu_blocks(
        allowed_cpus=allowed_cpu_list,
        n_phys_cores_per_run=n_phys_cores_per_run,
        use_smt=use_smt,
        socket_policy=socket_policy,
    )
    if not run_cpu_blocks:
        raise RuntimeError("No run slots could be allocated from allowed CPUs.")

    n_tasks = len(param_list)
    if n_tasks == 0:
        return [] if return_outputs else None

    n_slots = len(run_cpu_blocks)
    n_workers = min(n_slots, n_tasks)
    if max_parallel_runs is not None:
        n_workers = min(n_workers, int(max_parallel_runs))

    inject_n_jobs = _func_accepts_n_jobs(func)
    n_jobs_value = int(info["n_jobs_value"])

    if verbose:
        static_keys = 0 if static_params_local is None else len(static_params_local)
        print(
            f"[run_queue_pinned] tasks={n_tasks} | workers={n_workers}/{n_slots} | "
            f"allowed_cpus={info['allowed_cpu_count']} | sockets={info['sockets_visible']} | "
            f"threads_per_core={info['threads_per_core_visible']} | use_smt={info['use_smt_resolved']} | "
            f"socket_policy={info['socket_policy']} | "
            f"static_keys={static_keys} | "
            f"inject_n_jobs={inject_n_jobs} | n_jobs_value={n_jobs_value} | override_n_jobs={override_n_jobs}"
        )

    ctx = get_context(start_method)
    task_q = ctx.Queue(maxsize=2 * n_workers)
    out_q = ctx.Queue()

    procs = []
    for w in range(n_workers):
        p = ctx.Process(
            target=_worker_loop,
            args=(
                func,
                run_cpu_blocks[w],
                static_params_local,
                return_outputs,
                inject_n_jobs,
                n_jobs_value,
                override_n_jobs,
                set_thread_env_limits,
                pinning_policy,
                task_q,
                out_q,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # init progress bar
    if progress_bar and _TQDM_AVAILABLE:
        pbar = tqdm(
            total=n_tasks,
            desc="Processing tasks",
            unit="task",
            mininterval=0.1,
            miniters=1,
            position=progress_bar_position,
            leave=True,
        )
    else:
        if progress_bar and not _TQDM_AVAILABLE:
            print("[run_queue_pinned] Warning: tqdm not installed, progress bar disabled.", file=sys.stderr)
        pbar = None

    for i, kwargs in enumerate(param_list):
        task_q.put((i, kwargs))

    for _ in range(n_workers):
        task_q.put(None)

    results: Optional[List[Any]] = [None] * n_tasks if return_outputs else None
    errors: List[Tuple[int, str]] = []

    for _ in range(n_tasks):
        idx, res, tb = out_q.get()

        if tb is not None:
            errors.append((idx, tb))
        if results is not None:
            results[idx] = res
        if pbar is not None:
            pbar.update(1)
            sys.stdout.flush()

    if pbar is not None:
        pbar.close()

    for p in procs:
        p.join()

    if errors:
        errors.sort(key=lambda x: x[0])
        for idx, tb in errors:
            print(f"\n[ERROR task={idx}] traceback:\n{tb}")
        if raise_on_error:
            raise RuntimeError(f"{len(errors)} task(s) failed. See tracebacks above.")

    return results
