
import numpy as np

float_name = { np.single: 'single',
               np.double: 'double',
               np.complex64: 'single complex',
               np.complex128: 'double complex', }

from benchmark import Benchmark
from bench_gemm import BenchGEMM
from bench_cholesky import BenchCholesky
from bench_numexpr import BenchNumexpr, BenchNumexpr_Trig
from bench_fft import BenchFFT
