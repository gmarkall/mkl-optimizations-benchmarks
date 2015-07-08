
from __future__ import print_function

from benchmark import (BenchGEMM, BenchCholesky, BenchNumexpr,
                       BenchNumexpr_Trig, BenchFFT)
import numpy as np

def plat_info():
    import platform

    rv = [('platform', platform.platform()),
          ('processor', platform.processor())]
    try:
        import mkl

        rv.extend([('mkl', mkl.get_version_string()),
                   ('mkl-max-threads', str(mkl.get_max_threads())),
                   ('mkl-cpu-frequency', str(mkl.get_cpu_frequency())),
                   ('mkl-fft', str(np.fft.using_mklfft))])
    except ImportError:
        rv.extend([('mkl', 'No')])

    return rv


def setup_benchs():
    Ns = np.exp2(np.arange(6,13.5,0.5)).astype(np.int)
    FFT_Ns = np.exp2(np.arange(4,25)).astype(np.int)
    # BenchGEMM
    result = [
        BenchFFT([{'dtype': np.complex64, 'size': n} for n in FFT_Ns]),
        BenchGEMM([{'dtype': np.double, 'dims': (n,n,n)} for n in Ns]),
        BenchGEMM([{'dtype': np.single, 'dims': (n,n,n)} for n in Ns]),
        BenchCholesky([{'dtype': np.double, 'dim': n} for n in Ns]),
        BenchCholesky([{'dtype': np.single, 'dim': n} for n in Ns]),
        BenchNumexpr([{'dtype': np.double, 'dim': n*n} for n in Ns]),
        BenchNumexpr([{'dtype': np.single, 'dim': n*n} for n in Ns]),
        BenchNumexpr_Trig([{'dtype': np.double, 'dim': n*n} for n in Ns]),
        BenchNumexpr_Trig([{'dtype': np.single, 'dim': n*n} for n in Ns]),
    ]
    return result

def main():
    print('\n'.join([': '.join(a) for a in plat_info()]))
    benchs_to_run = setup_benchs()

    for bench in benchs_to_run:
        for e, time in bench.profile_all():
            name, estimate, units = bench.name(e), bench.estimate(e), bench.units(e)
            tp = estimate / time
            time *= 1000
            print("{name}: {time:8.3f}ms ({tp:6.3f} {units})".format(**locals()))


if __name__=='__main__':
    main()
