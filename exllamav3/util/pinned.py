"""
Exact-size page-locked host buffers that are reused by size. torch's pinned allocator rounds every
block up to a power of two (a 671 MB state takes 1 GiB, a 1.02 GB row of logits 2 GiB), which
nearly halves the host memory a job can use for staged tensors.
"""

import math
import mmap
import threading
import weakref
from collections import defaultdict

import torch


class _LockedMemory(mmap.mmap):
    """Anonymous memory registered with the CUDA driver (page-locked) for as long as it lives.
    torch.frombuffer keeps it alive while any tensor views it; it is unregistered just before it is
    unmapped"""
    ptr = None

    def __del__(self):
        if self.ptr is not None:
            torch.cuda.cudart().cudaHostUnregister(self.ptr)


class PinnedPool:
    """Exact-size page-locked host buffers, handed out again by size once given back. torch's
    pinned allocator rounds every block up to a power of two, so a 671 MB state takes 1 GiB of
    locked RAM and a 1.02 GB row of reference logits takes 2 GiB: nearly half the RAM a pass could
    use. Buffers nobody gives back are freed as usual when their last view goes. Pageable (only
    slower to copy) if registering fails.

    Registering costs about 1 s per GB (the pages are faulted in and locked), so reserve() can
    make buffers ahead of need in a background thread"""

    def __init__(self, enabled):
        self.enabled = enabled
        self.free = defaultdict(list)                # nbytes -> buffers given back or reserved
        self.live = weakref.WeakValueDictionary()    # data_ptr -> buffer
        self.pending = defaultdict(int)              # nbytes -> buffers a reserve() thread will add
        self.cond = threading.Condition()

    HUGE = True    # back buffers with transparent huge pages when the kernel allows

    def _make(self, n):
        mem = _LockedMemory(-1, n)
        if self.HUGE and hasattr(mmap, "MADV_HUGEPAGE"):
            # Registering faults in and locks every page; 2 MB pages make that far fewer faults
            try:
                mem.madvise(mmap.MADV_HUGEPAGE)
            except OSError:
                pass
        ptr = torch.frombuffer(mem, dtype = torch.uint8).data_ptr()
        try:
            cudart = torch.cuda.cudart()
            ok = cudart.cudaHostRegister(ptr, n, 0) == cudart.cudaError.success
        except Exception:
            ok = False
        if not ok:
            return None
        mem.ptr = ptr
        return mem

    def empty(self, shape, dtype):
        n = math.prod(shape) * dtype.itemsize
        if not self.enabled or n == 0:
            return torch.empty(shape, dtype = dtype)
        with self.cond:
            # A buffer of this size that a reserve() thread is making is waited for, not duplicated
            while not self.free[n] and self.pending[n] > 0:
                self.cond.wait()
            mem = self.free[n].pop() if self.free[n] else None
        if mem is None:
            mem = self._make(n)
            if mem is None:
                self.enabled = False
                return torch.empty(shape, dtype = dtype)
            with self.cond:
                self.live[mem.ptr] = mem
        return torch.frombuffer(mem, dtype = torch.uint8).view(dtype).view(shape)

    def reserve(self, nbytes, count):
        """Make up to count buffers of nbytes in a background thread, beyond those already free"""
        if not self.enabled or nbytes == 0:
            return
        with self.cond:
            count -= len(self.free[nbytes]) + self.pending[nbytes]
            if count <= 0:
                return
            self.pending[nbytes] += count

        def work():
            for k in range(count):
                mem = self._make(nbytes) if self.enabled else None
                with self.cond:
                    self.pending[nbytes] -= 1
                    if mem is None:
                        # Give up on the rest; empty() falls back to its own path
                        self.pending[nbytes] -= count - k - 1
                        self.cond.notify_all()
                        return
                    self.live[mem.ptr] = mem
                    self.free[nbytes].append(mem)
                    self.cond.notify_all()

        threading.Thread(target = work, daemon = True).start()

    def give(self, tensors, keep = None):
        """Return whole buffers from empty() for reuse (anything else is ignored). keep: hold at
        most this many free buffers of a size; the rest are released when their views go. The
        caller guarantees no unfinished copy still uses them, or that later copies into them are
        ordered after it on the same streams"""
        with self.cond:
            for t in tensors if isinstance(tensors, (list, tuple)) else ():
                mem = self.live.get(t.data_ptr()) if t.device.type == "cpu" else None
                if mem is not None and t.numel() * t.element_size() == len(mem):
                    if keep is None or len(self.free[len(mem)]) < keep:
                        self.free[len(mem)].append(mem)
            self.cond.notify_all()
