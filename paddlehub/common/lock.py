# coding:utf8
import threading
import time
from collections import deque
from itertools import islice
from sys import version_info
if version_info.major == 2:
    import thread as _thread
elif version_info.major == 3:
    import _thread


class Condition:
    def __init__(self, lock=None):
        if lock is None:
            lock = _thread.RLock()
        self._lock = lock
        self.acquire = lock.acquire
        self.release = lock.release
        try:
            self._release_save = lock._release_save
        except AttributeError:
            pass
        try:
            self._acquire_restore = lock._acquire_restore
        except AttributeError:
            pass
        try:
            self._is_owned = lock._is_owned
        except AttributeError:
            pass
        self._waiters = deque()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)

    def __repr__(self):
        return "<Condition(%s, %d)>" % (self._lock, len(self._waiters))

    def _release_save(self):
        self._lock.release()

    def _acquire_restore(self, x):
        self._lock.acquire()

    def _is_owned(self):
        if self._lock.acquire(0):
            self._lock.release()
            return False
        else:
            return True

    def wait(self, timeout=None):
        if not self._is_owned():
            print("cannot wait on un-acquired lock")
            return False
        waiter = _thread.allocate_lock()
        waiter.acquire()
        self._waiters.append(waiter)
        saved_state = self._release_save()
        gotit = False
        try:
            if timeout is None:
                waiter.acquire()
                gotit = True
            else:
                if timeout > 0:
                    gotit = waiter.acquire(True, timeout)
                else:
                    gotit = waiter.acquire(False)
            return gotit
        finally:
            self._acquire_restore(saved_state)
            if not gotit:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass

    def wait_for(self, predicate, timeout=None):
        endtime = None
        waittime = timeout
        result = predicate()
        while not result:
            if waittime is not None:
                if endtime is None:
                    endtime = time.monotonic() + waittime
                else:
                    waittime = endtime - time.monotonic()
                    if waittime <= 0:
                        break
            self.wait(waittime)
            result = predicate()
        return result

    def notify(self, n=1):
        if not self._is_owned():
            print("cannot notify on un-acquired lock")
            return False
        all_waiters = self._waiters
        waiters_to_notify = deque(islice(all_waiters, n))
        if not waiters_to_notify:
            return
        for waiter in waiters_to_notify:
            waiter.release()
            try:
                all_waiters.remove(waiter)
            except ValueError:
                pass

    def notify_all(self):
        self.notify(len(self._waiters))


class RWLock(object):
    def __init__(self):
        self.lock = _thread.allocate_lock()
        self.read_cond = Condition(self.lock)
        self.write_cond = Condition(self.lock)
        self.read_waiter = 0
        self.write_waiter = 0
        self.state = 0
        self.owners = []

    def write_acquire(self, blocking=True):
        me = _thread.get_ident()
        with self.lock:
            while not self._write_acquire(me):
                if not blocking:
                    return False
                self.write_waiter += 1
                self.write_cond.wait()
                self.write_waiter -= 1
        return True

    def _write_acquire(self, me):
        # 获取写锁只有当锁没人占用，或者当前线程已经占用
        if self.state == 0 or (self.state < 0 and me in self.owners):
            self.state -= 1
            self.owners.append(me)
            return True
        if self.state > 0 and me in self.owners:
            print('cannot recursively wrlock a rdlocked lock')
            return False
        return False

    def read_acquire(self, blocking=True):
        me = _thread.get_ident()
        with self.lock:
            while not self._read_acquire(me):
                if not blocking:
                    return False
                self.read_waiter += 1
                self.read_cond.wait()
                self.read_waiter -= 1
        return True

    def _read_acquire(self, me):
        if self.state < 0:
            # 如果锁被写锁占用
            return False

        if not self.write_waiter:
            ok = True
        else:
            ok = me in self.owners
        if ok:
            self.state += 1
            self.owners.append(me)
            return True
        return False

    def unlock(self):
        me = _thread.get_ident()
        with self.lock:
            try:
                self.owners.remove(me)
            except ValueError:
                print('cannot release un-acquired lock')
                return False
            if self.state > 0:
                self.state -= 1
            else:
                self.state += 1
            if not self.state:
                if self.write_waiter:  # 如果有写操作在等待（默认写优先）
                    self.write_cond.notify()
                elif self.read_waiter:
                    self.read_cond.notify_all()
                elif self.write_waiter:
                    self.write_cond.notify()

    read_release = unlock
    write_release = unlock


lock = RWLock()
