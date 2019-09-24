import os
if os.name == "posix":
    import fcntl


class WinLock(object):
    def flock(self, *args):
        pass

    def __init__(self):
        self.LOCK_EX = ""
        self.LOCK_UN = ""


class Lock(object):
    _owner = None

    def __init__(self):
        if os.name == "posix":
            self.lock = fcntl
        else:
            self.lock = WinLock()
        _lock = self.lock
        self.LOCK_EX = self.lock.LOCK_EX
        self.LOCK_UN = self.lock.LOCK_UN

    def get_lock(self):
        return self.lock

    def flock(self, fp, cmd):
        if cmd == self.lock.LOCK_UN:
            Lock._owner = None
            self.lock.flock(fp, cmd)
        elif cmd == self.lock.LOCK_EX:
            if Lock._owner is None:
                Lock._owner = os.getpid()
                self.lock.flock(fp, cmd)
            else:
                if Lock._owner == os.getpid():
                    pass
                else:
                    self.lock.flock(fp, cmd)


lock = Lock()
