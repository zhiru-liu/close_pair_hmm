import cphmm.recomb_inference as ri
import cphmm.cphmm as cphmm
import infer_pipelines

import numpy as np
import time


def test_performance():
    # generate a random boolean sequence

    np.random.seed(0)
    n = 200000
    seq = np.zeros(n, dtype=bool)
    seq[5000:10000] = 1

    # init a model
    model = infer_pipelines.init_hmm('Bacteroides_vulgatus_57955', n, cphmm.config.HMM_BLOCK_SIZE)

    # run the inference
    start = time.time()
    for i in range(1):
        _ = ri.infer(seq, np.zeros(n), model, cphmm.config.HMM_BLOCK_SIZE, clade_cutoff_bin=40)
    end = time.time()
    print('Time elapsed: ', end - start)


if __name__ == '__main__':
    test_performance()