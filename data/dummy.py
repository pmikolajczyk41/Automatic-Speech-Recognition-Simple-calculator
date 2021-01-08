from random import randint
from typing import List, Union

import numpy as np
from scipy.stats import multivariate_normal

from data import TRAIN_DIR


def generate_dummy_mfccs(nfiles: int,
                         nframes: int, nframes_delta: int,
                         state_durations: List[float],
                         means: List[List[float]],
                         stdevs: Union[List[List[float]], List[int]]) -> None:
    for file_id in range(nfiles):
        n = randint(nframes - nframes_delta, nframes + nframes_delta)
        state_emissions = [int(sd * n) for sd in state_durations]

        observations = [multivariate_normal.rvs(mean, stdev, size=se)
                        for se, mean, stdev in zip(state_emissions, means, stdevs)]
        np.savetxt(TRAIN_DIR / f'dummy({file_id}).mfcc', np.vstack(observations))


if __name__ == '__main__':
    generate_dummy_mfccs(5,
                         15, 2,
                         [0.3, 0.2, 0.1, 0.4],
                         [[-10, 0], [-5, 0], [0, 0], [5, 0]], 4 * [1])
