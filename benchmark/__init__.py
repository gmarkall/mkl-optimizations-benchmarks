
import numpy as np

float_name = { np.single: 'single',
               np.double: 'double' }

from benchmark import Benchmark
from bench_gemm import BenchGEMM
from bench_cholesky import BenchCholesky
from bench_numexpr import BenchNumexpr, BenchNumexpr_Trig
