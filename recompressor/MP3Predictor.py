import numpy as np


class MP3Predictor:

    def __init__(self, n_samples):
        self.n_regions = 20
        self.region_size = n_samples // self.n_regions
        self.counter = np.zeros((self.n_regions, 512, 2), dtype=int)  # bitcounts per byte and per sample

    def get_region(self, sample_idx):
        upper = self.region_size
        for i in range(self.n_regions):
            if sample_idx < upper:
                return i
            upper += self.region_size
        return self.n_regions - 1

    def p(self, sample_idx, context):
        return np.int64(4096 * (self.counter[self.get_region(sample_idx), context, 1] + 1) / (
                self.counter[self.get_region(sample_idx), context, 0] + self.counter[self.get_region(sample_idx), context, 1] + 2))

    def update(self, y, sample_idx, context):
        self.counter[self.get_region(sample_idx), context, y] += 1
        if self.counter[self.get_region(sample_idx), context, y] > 65534:
            self.counter[self.get_region(sample_idx), context, 0] >>= 1
            self.counter[self.get_region(sample_idx), context, 1] >>= 1
        # self.context += self.context + y # done by caller
        # if self.context >= 256:
        #     self.context = 1
