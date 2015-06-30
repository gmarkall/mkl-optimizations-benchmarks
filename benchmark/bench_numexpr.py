from __future__ import print_function, absolute_import

from benchmark import Benchmark, float_name
import numpy as np
import numexpr as ne

class BenchNumexpr(Benchmark):
    def setup(self, experiment):
        dtype = experiment['dtype']
        N = experiment['dim']
        self.x = np.asarray(np.linspace(-1, 1, N), dtype=dtype)
        self.y = np.asarray(np.linspace(-1, 1, N), dtype=dtype)
        self.z = np.empty_like(self.x)
        
    def teardown(self, experiment):
        del(self.z)
        del(self.y)
        del(self.x)

    def run(self, experiment):
        x, y, z = self.x, self.y, self.z
        ne.evaluate('2*y+4*x', out=z)

    def estimate(self, experiment):
        dtype = experiment['dtype']
        N = experiment['dim']
        return 3 * dtype().itemsize * N * 1e-9

    def units(self, experiment):
        return 'GBytes/s'

    def name(self, experiment):
        name = float_name[experiment['dtype']]
        N = experiment['dim']
        return 'Numexpr 2*y+4*x [{name}] (N={N:5d})'.format(**locals())

    
class BenchNumexpr_Trig(Benchmark):
    def setup(self, experiment):
        dtype = experiment['dtype']
        N = experiment['dim']
        self.x = np.asarray(np.linspace(-1, 1, N), dtype=dtype)
        self.y = np.asarray(np.linspace(-1, 1, N), dtype=dtype)
        self.z = np.empty_like(self.x)
        
    def teardown(self, experiment):
        del(self.z)
        del(self.y)
        del(self.x)

    def run(self, experiment):
        x, y, z = self.x, self.y, self.z
        ne.evaluate('cos(y)*sin(x)+cos(x)*sin(y)', out=z)

    def estimate(self, experiment):
        dtype = experiment['dtype']
        N = experiment['dim']
        return 3 * dtype().itemsize * N * 1e-9

    def units(self, experiment):
        return 'GBytes/s'

    def name(self, experiment):
        name = float_name[experiment['dtype']]
        N = experiment['dim']
        return 'Numexpr cos(y)*sin(x)+cos(x)*sin(y) [{name}] (N={N:5d})'.format(**locals())
