#!/usr/bin/env python3
import fcntl
import time
from pathlib import Path
import logging
from typing import Self

logger = logging.getLogger(__name__)


class DirGuard:
    """
    Context manager that guards a directory from concurrent access.
    Blocks until a lock file in the specified directory is acquired.

    Provides convenience functions to get/set the creation time of the path
    to determine if the data within is stale (and take user-defined actions)
    """

    def __init__(
        self,
        path,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.timestamp = self.path / '.CREATION_TIME'
        self.timestamp.touch(exist_ok=True)
        self.lock = self.path.with_name(f'.{self.path.name}.lock')

    def __enter__(self) -> Self:
        logger.debug(f'Acquiring lock on {self.lock}')
        self.lockfile = open(self.lock, 'a+')
        fcntl.flock(self.lockfile.fileno(), fcntl.LOCK_EX)
        logger.debug(f'Acquired lock on {self.lock}')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.lockfile.close()  # This releases the lock
        logger.debug(f'Released lock on {self.lock}')

    def update_creation_timestamp(self):
        t = int(time.time())
        self.timestamp.write_text(str(t))
        logger.debug(f'Updated timestamp {self.timestamp} to {t}')

    def days_since_creation(self) -> float:
        try:
            creation_time = int(self.timestamp.read_text())
            return (time.time() - creation_time) / 86400
        except:
            return float('inf')
