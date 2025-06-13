"""
add by kexinchu - for global epoch(time_windows)
"""

import time
import threading
from typing import Optional

# global setting for the length of time_window
TIME_WINDOW_LENGTH = 10 # seconds

class GlobalEpoch:
    _instance: Optional['GlobalEpoch'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self._epoch = 0
        self._last_update = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._update_epoch, daemon=True)
        self._thread.start()
    
    def _update_epoch(self):
        while self._running:
            current_time = time.time()
            if current_time - self._last_update >= TIME_WINDOW_LENGTH:
                self._epoch += 1
                self._last_update = current_time
            time.sleep(0.1)  # Small sleep to prevent CPU overuse
    
    @property
    def epoch(self) -> int:
        return self._epoch
    
    def stop(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join()

# Create a global instance
global_epoch = GlobalEpoch()

def get_epoch() -> int:
    """Get the current global epoch value."""
    return global_epoch.epoch 