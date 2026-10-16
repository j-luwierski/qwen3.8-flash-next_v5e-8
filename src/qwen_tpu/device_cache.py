"""Bounded device weights: protect small frequently reused norms/convolutions.

Bulk admission is sticky, not cyclic LRU: streaming through all layers must not
flush every resident matrix before the next autoregressive token uses it.
"""

from collections import OrderedDict


class DeviceWeightCache:
    def __init__(self, budget_mib=0, small_mib=32, small_limit=65536):
        if budget_mib < 0 or small_mib < 0:
            raise ValueError('Weight cache budgets must be nonnegative')
        self.budget = int(budget_mib * 1024 ** 2)
        self.small_budget = min(int(small_mib * 1024 ** 2), self.budget)
        self.small_limit = small_limit
        self.small, self.bulk = OrderedDict(), {}
        self.small_bytes = self.bulk_bytes = 0
        self.hits = self.misses = self.host_bytes = 0

    def get(self, key):
        if key in self.small:
            self.small.move_to_end(key)
            self.hits += 1
            return self.small[key]
        if key in self.bulk:
            self.hits += 1
            return self.bulk[key]
        self.misses += 1
        return None

    def put(self, key, value):
        size = value.size * value.dtype.itemsize
        self.host_bytes += size
        if not self.budget:
            return value
        if size <= self.small_limit and size <= self.small_budget:
            while self.small and self.small_bytes + size > self.small_budget:
                _, old = self.small.popitem(last=False)
                self.small_bytes -= old.size * old.dtype.itemsize
            self.small[key] = value
            self.small_bytes += size
        elif self.bulk_bytes + size <= self.budget - self.small_budget:
            self.bulk[key] = value
            self.bulk_bytes += size
        return value

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, host_to_device_weight_bytes=self.host_bytes,
                    resident_bytes=self.small_bytes + self.bulk_bytes, budget_bytes=self.budget)

    def clear(self):
        self.small.clear()
        self.bulk.clear()
        self.small_bytes = self.bulk_bytes = 0
