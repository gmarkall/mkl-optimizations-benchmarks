from __future__ import print_function, absolute_import

from benchmark import Benchmark, float_name
import numpy as np

class BenchGEMM(Benchmark):
    def setup(self, experiment):
        M, N, K = experiment['dims']
        dtype = experiment['dtype']
        self.A = np.asarray(np.random.rand(M, K), dtype=dtype, order='F')
        self.B = np.asarray(np.random.rand(K, N), dtype=dtype, order='F')

    def teardown(self, experiment):
        del(self.B)
        del(self.A)

    def run(self, experiment):
        return self.A.dot(self.B)

    def estimate(self, experiment):
        M, N, K = experiment['dims']
        return (M*N*(K+2) + M*N*K) * 1e-9

    def units(self, experiment):
        return "GFLOPS ({0})".format(float_name[experiment['dtype']])

    def name(self, experiment):
        m, n, k = experiment['dims']
        name_map = { np.double: 'dgemm',
                     np.single: 'sgemm' }
        name = name_map[experiment['dtype']]
        
        return "{name}(M={m:5d}, N={n:5d}, K={k:5d})".format(**locals())

