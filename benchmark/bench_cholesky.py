from __future__ import print_function, absolute_import

from benchmark import Benchmark, float_name
import numpy as np

class BenchCholesky(Benchmark):
    def setup(self, experiment):
        M = experiment['dim']
        dtype = experiment['dtype']
        A = np.asarray(np.random.rand(M, M), dtype=dtype)
        self.A = A * A.transpose() + M*np.eye(M)

    def teardown(self, experiment):
        del(self.A)

    def run(self, experiment):
        return np.linalg.cholesky(self.A)

    def estimate(self, experiment):
        """taken from http://www.cs.utexas.edu/users/flame/Notes/NotesOnCholReal.pdf"""
        M = experiment['dim']
        return (M*M*M/3.0 + M*M/2.0) * 1e-9

    def units(self, experiment):
        if experiment['dtype'] is np.double:
            return "GFLOPS (double)"
        else:
            return "GFLOPS (single)"

    def name(self, experiment):
        M = experiment['dim']
        name_map = { np.double: 'dpotrf',
                     np.single: 'spotrf' }
        lapack_func = name_map[experiment['dtype']]
        return "cholesky [{lapack_func}] (M={M:5d})".format(**locals())
