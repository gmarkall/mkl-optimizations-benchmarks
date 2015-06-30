"""A runner module that runs the bench in throwaway environments, both with mkl
and without.

Note this requires a quite up-to-date "conda_api", that right now can only be
installed from the repo.

"""
from __future__ import print_function

import os
import subprocess
import contextlib
import tempfile
import shutil
import json

import conda_api


test_packages = [
    'scipy',
    'scikit-learn',
    'numexpr',
    'pymc',
    ]

mkl_packages = [
    'mkl',
    'mkl-service',
    'mklfft',
]

@contextlib.contextmanager
def anonymous_environment(packages):
    base_path = tempfile.mkdtemp(prefix='tmp-mkl-runner-')
    env_path = os.path.join(base_path, 'environment') 
    conda_api.create(path=env_path, pkgs=packages)
    yield env_path
    conda_api.remove_environment(prefix=env_path)
    shutil.rmtree(base_path)

def get_conda_root_prefix():
    info = subprocess.check_output(['conda', 'info', '--json']) 
    info = json.loads(info.decode())
    return info['root_prefix']

def main():
    conda_prefix = get_conda_root_prefix()
    conda_api.set_root_prefix(prefix=conda_prefix)
    
    print(" CREATING ENV W/O MKL ".center(80, '='))
    with anonymous_environment(test_packages) as env:
        print(" RUNNING BENCH ".center(80, '='))
        conda_api.process(path=env, cmd='python', args=['run-bench.py']).wait()
        
    print(" CREATING ENV WITH MKL ".center(80, '='))
    with anonymous_environment(test_packages + mkl_packages) as env:
        print(" RUNNING BENCH ".center(80, '='))
        conda_api.process(path=env, cmd='python', args=['run-bench.py']).wait()
        

if __name__=='__main__':
    main()
