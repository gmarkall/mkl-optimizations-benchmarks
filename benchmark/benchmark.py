
from __future__ import print_function
import timeit
import gc

class Benchmark(object):
    """A benchmark will need to provide the following methods:

    - setup: create data for the experiment
    - teardown: dispose of data for the experiment
    - run: run the experiment

    - estimate: return an estimate of the operations performed
    - units: unit for the estimation (when divided by seconds)
    """
    _trials = 3

    def __init__(self, experiments):
        self.experiments = experiments

    def profile(self, experiment):
        self.setup(experiment)
        timer = timeit.default_timer
        
        gcold = gc.isenabled()
        gc.disable()

        times = []
        for i in xrange(self._trials):
            t = timer()
            self.run(experiment)
            t = timer() - t
            times.append(t)
            
        if gcold:
            gc.enable()
        
        self.teardown(experiment)
        return experiment, min(times)

    def profile_all(self):
        for e in self.experiments:
            yield self.profile(e)
