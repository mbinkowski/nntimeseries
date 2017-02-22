from __init__ import *
from artificial_data_utils import *

SS = []
for n in [100000]:
    for ss in [False]:
        for sources in [64]:
            SS.append(NoisySignal(exponential_time=True, single_source=ss,
 		                  filepath=True, n=n, sources=sources))
