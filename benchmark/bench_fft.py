from __future__ import print_function, absolute_import

from benchmark import Benchmark, float_name
import numpy as np

class BenchFFT(Benchmark):
    def setup(self, experiment):
        size = experiment['size']
        dtype = experiment['dtype']
        a = np.random.randn(size) + 1j * np.random.randn(size)
        self.a = a.astype(dtype)

    def teardown(self, experiment):
        del(self.a)

    def run(self, experiment):
        return np.fft.fftn(self.a)

    def estimate(self, experiment):
        size = experiment['size']
        return 5.0 * size * np.log2(size) * 1e-9

    def units(self, experiment):
        return 'GFLOPS ({0})'.format(float_name[experiment['dtype']])

    def name(self, experiment):
        size = experiment['size']
        return 'fft (size={size:5d})'.format(**locals())
