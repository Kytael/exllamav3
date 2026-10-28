"""
Exact-size page-locked host buffers that are reused by size. torch's pinned allocator rounds every
block up to a power of two (a 671 MB state takes 1 GiB, a 1.02 GB row of logits 2 GiB), which
nearly halves the host memory a job can use for staged tensors.
"""

import math
import mmap
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
    slower to copy) if registering fails"""

    def __init__(self, enabled):
        self.enabled = enabled
        self.free = defaultdict(list)                # nbytes -> buffers given back
        self.live = weakref.WeakValueDictionary()    # data_ptr -> buffer

    def empty(self, shape, dtype):
        n = math.prod(shape) * dtype.itemsize
        if not self.enabled or n == 0:
            return torch.empty(shape, dtype = dtype)
        if self.free[n]:
            mem = self.free[n].pop()
        else:
            mem = _LockedMemory(-1, n)
            ptr = torch.frombuffer(mem, dtype = torch.uint8).data_ptr()
            try:
                cudart = torch.cuda.cudart()
                ok = cudart.cudaHostRegister(ptr, n, 0) == cudart.cudaError.success
            except Exception:
                ok = False
            if not ok:
                self.enabled = False
                return torch.empty(shape, dtype = dtype)
            mem.ptr = ptr
            self.live[ptr] = mem
        return torch.frombuffer(mem, dtype = torch.uint8).view(dtype).view(shape)

    def give(self, tensors):
        """Return whole buffers from empty() for reuse (anything else is ignored). The caller
        guarantees no unfinished copy still uses them, or that later copies into them are ordered
        after it on the same streams"""
        for t in tensors if isinstance(tensors, (list, tuple)) else ():
            mem = self.live.get(t.data_ptr()) if t.device.type == "cpu" else None
            if mem is not None and t.numel() * t.element_size() == len(mem):
                self.free[len(mem)].append(mem)
