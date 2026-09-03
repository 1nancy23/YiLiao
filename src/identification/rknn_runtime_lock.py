import threading


_LOCKS = {}
_LOCKS_GUARD = threading.Lock()
_SECONDARY_NPU_LOCK = threading.RLock()


def get_rknn_lock(core_mask, secondary_domain=False):
    if secondary_domain:
        return _SECONDARY_NPU_LOCK
    key = int(core_mask)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock
