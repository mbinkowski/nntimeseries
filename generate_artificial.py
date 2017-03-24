from nnts.artificial import NoisySignal

SS = []
for n in [10000]:
    for ss in [True, False]:
        for sources in [16, 64]:
            SS.append(NoisySignal(exponential_time=True, single_source=ss,
 		                  filepath=True, n=n, sources=sources))
