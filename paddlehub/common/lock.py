import fcntl
import os


class WinLock(object):
    def flock(self, *args):
        pass

    def __init__(self):
        self.LOCK_EX = ""
        self.LOCK_UN = ""
        self.LOCK_TE = ""


class Lock(object):
    def __init__(self):
        if os.name == "posix":
            self.lock = fcntl
        else:
            self.lock = WinLock()

    def get_lock(self):
        return self.lock


lock = Lock().get_lock()
