import heapq
import itertools
from typing import Generic, List, Tuple, TypeVar

T = TypeVar("T")


class TopN(Generic[T]):
    """
    Keep the top-N items by score (highest scores). Tie-breaking is handled
    by a monotonically increasing counter so items (which may be unorderable)
    are never compared directly.

    Args:
        n: how many top items to keep (N).
        prefer_newer: when a new item has the same score as the current min,
                      if True the new item will replace the old one; if False
                      the old item is kept (stable, first-seen wins).
    """

    def __init__(self, n: int, prefer_newer: bool = False):
        if n <= 0:
            raise ValueError("n must be > 0")
        self.n = n
        self.heap: List[Tuple[float, int, T]] = []
        self._counter = itertools.count()
        self.prefer_newer = prefer_newer

    def push(self, score: float, element: T) -> None:
        """Consider (score, element) for the top-N set."""
        entry = (score, next(self._counter), element)
        if len(self.heap) < self.n:
            heapq.heappush(self.heap, entry)
            return

        # heap[0] is the smallest (score, counter, element)
        min_score, _, _ = self.heap[0]

        if score > min_score:
            # strictly better score -> replace
            heapq.heapreplace(self.heap, entry)
        elif self.prefer_newer and score == min_score:
            # equal score but we prefer newer items: replace oldest-equal
            heapq.heapreplace(self.heap, entry)
        # else: score <= min_score and prefer_newer==False -> do nothing

    def get_elements(self) -> List[T]:
        """Return elements sorted by descending score (highest first)."""
        # sort by score then counter (both present in heap entries)
        return [elem for (score, _, elem) in sorted(self.heap, reverse=True)]

    def get_items(self) -> List[Tuple[float, T]]:
        """Return (score, element) pairs sorted by descending score."""
        return [(score, elem) for (score, _, elem) in sorted(self.heap, reverse=True)]

    def __len__(self) -> int:
        return len(self.heap)

    def clear(self) -> None:
        self.heap.clear()
        self._counter = itertools.count()
