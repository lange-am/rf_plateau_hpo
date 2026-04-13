"""
file_mover.py

Module for background moving of "old" files from a temporary directory (e.g., /tmp)
to persistent storage. Designed for use in Jupyter notebooks.

Main components:
- FileMoverThread: a thread that periodically scans the source directory and moves files
  that have not been modified for at least `min_age` seconds, optionally ignoring files
  with a specific extension (e.g., '.tmp').
- clean_source_dir(): deletes all contents of the source directory (use with caution!).
"""

import os
import sys
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

try:
    from tqdm.auto import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    tqdm = None


class FileMoverThread(threading.Thread):
    """
    Background thread that moves files from source_dir to dest_dir.

    The thread checks all files in source_dir (recursively) every `check_interval` seconds.
    A file is moved if its age (time since last modification) is >= `min_age` and it is not
    ignored (i.e., its extension does not match `ignore_ext`).

    Args:
        source_dir (str or Path): source directory to monitor.
        dest_dir (str or Path): destination directory for moved files.
        min_age (int): minimum file age in seconds (default: 60).
        check_interval (int): interval between scans in seconds (default: 30).
        ignore_ext (str, optional): file extension to ignore (e.g., '.tmp'). Files with this
            extension will never be moved. Default is None.
        autostart (bool): if True, the thread starts immediately (default: False).
    """

    def __init__(
        self, 
        source_dir, 
        dest_dir, 
        min_age=60, 
        check_interval=30,
        ignore_ext=None, 
        total_files=None, 
        progress_bar=False, 
        progress_bar_position=None,
        autostart=False,
        verbose=0,
    ):
        super().__init__()
        self.source = Path(source_dir).resolve()
        self.dest = Path(dest_dir).resolve()
        self.min_age = min_age
        self.check_interval = check_interval
        self.ignore_ext = ignore_ext
        self.total_files = total_files
        self.progress_bar = progress_bar
        self.progress_bar_position = progress_bar_position
        self._stop_event = threading.Event()
        self.moved_count = 0
        self._moved_lock = threading.Lock()
        self._pbar = None
        self.daemon = True
        self.verbose = verbose

        if autostart:
            self.start()

    def run(self):
        if self.verbose > 0:
            print(f"[FileMover] Started monitoring {self.source} -> {self.dest} "
                  f"(min_age={self.min_age}s, ignore_ext={self.ignore_ext}, total_files={self.total_files})")
        self.dest.mkdir(parents=True, exist_ok=True)

        if _TQDM_AVAILABLE and self.progress_bar and self.total_files is not None:
            self._pbar = tqdm(
                total=self.total_files, 
                desc="Moving files", 
                unit="file", 
                miniters=1, 
                position=self.progress_bar_position,
                leave=True,
            )

        while not self._stop_event.is_set():
            try:
                self.move_old_files()
            except Exception as e:
                if self.verbose > 0:
                    print(f"[FileMover] Error in worker: {e}")
            self._stop_event.wait(self.check_interval)

        if self._pbar is not None:
            self._pbar.close()
        if self.verbose > 0:
            print("[FileMover] Stopped.")

    def move_old_files(self):
        """Scan source and move old files, ignoring those with ignore_ext."""
        now = time.time()
        for root, dirs, files in os.walk(self.source):
            for file in files:
                file_path = Path(root) / file

                # Skip files with the ignored extension
                if self.ignore_ext is not None and file.endswith(self.ignore_ext):
                    continue

                try:
                    # Check file age
                    mtime = file_path.stat().st_mtime
                    if now - mtime < self.min_age:
                        continue

                    # Relative path to preserve directory structure
                    rel_path = file_path.relative_to(self.source)
                    dest_path = self.dest / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    # Move the file
                    shutil.move(str(file_path), str(dest_path))
                    with self._moved_lock:
                        self.moved_count += 1
                        if self._pbar is not None:
                            self._pbar.update(1)
                            sys.stdout.flush()

                    if self.verbose > 1:
                        print(f"[FileMover] Moved: {file_path} -> {dest_path}")

                except (OSError, PermissionError):
                    # File might be in use or already deleted – ignore
                    continue

    def stop(self, timeout: Optional[float] = None):
        """
        Stop the thread and wait for it to finish.

        Args:
            timeout: maximum wait time in seconds (None = infinite).
        """
        self._stop_event.set()
        self.join(timeout)

    def wait_for_completion(
        self, target: Optional[int] = None, 
        timeout: Optional[float] = None
    ) -> int:
        """
        Wait until at least `target` files have been moved, or until timeout expires.

        Args:
            target: Number of files to wait for. If None, uses self.total_files.
            timeout: Maximum time to wait in seconds. None means wait indefinitely.

        Returns:
            The actual number of files moved at the moment of return.
        """
        if target is None:
            if self.total_files is None:
                raise ValueError("No target specified and total_files is not set.")
            target = self.total_files

        deadline = None if timeout is None else time.time() + timeout
        while True:
            current = self.moved_count
            if current >= target:
                if self.verbose > 0:
                    print(f"[FileMover] All {current} files moved")
                return current
            if deadline is not None and time.time() >= deadline:
                if self.verbose > 0:
                    print(f"[FileMover] Only {current} files out of {target} moved...")
                return current
            time.sleep(0.1)

def clean_source_dir(source_dir, confirm: bool = True, verbose: int = 0):
    """
    Delete all contents of the source directory. Use with caution!

    Args:
        source_dir (str or Path): directory to clean.
        confirm (bool): if True, ask for confirmation in the console.

    Returns:
        bool: True if cleaning was performed, False otherwise.
    """
    path = Path(source_dir).resolve()
    if not path.exists():
        print(f"[clean] Source directory {path} does not exist. Nothing to clean.")
        return False

    if confirm:
        answer = input(f"Are you sure you want to DELETE ALL contents of {path}? (y/N): ")
        if answer.lower() != 'y':
            print("[clean] Aborted.")
            return False

    try:
        shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        if verbose > 0:
            print(f"[clean] Removed and recreated {path}")
        return True
    except Exception as e:
        print(f"[clean] Error cleaning {path}: {e}")
        return False


# Example usage if run as a script
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python file_mover.py <source_dir> <dest_dir> [min_age] [check_interval] [ignore_ext]")
        sys.exit(1)

    source = sys.argv[1]
    dest = sys.argv[2]
    min_age = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    interval = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    ignore_ext = sys.argv[5] if len(sys.argv) > 5 else None

    mover = FileMoverThread(source, dest, min_age, interval, ignore_ext, autostart=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        mover.stop()
        print("Exiting.")