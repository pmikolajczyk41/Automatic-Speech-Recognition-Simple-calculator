from copy import deepcopy
from statistics import mean, stdev
from typing import NamedTuple, List, Optional

from scipy.stats import multivariate_normal

from data import FeatVec


class Distribution(NamedTuple):
    means: List[float]
    variances: List[float]


DIMENSIONALITY = 39
default_distribution = Distribution(DIMENSIONALITY * [0.], DIMENSIONALITY * [1.])


class State:
    def __init__(self, initial_distribution: Optional[Distribution] = None, loop: bool = False):
        if initial_distribution is not None:
            self.is_emitting, (self._means, self._vars) = True, deepcopy(initial_distribution)
        else:
            self.is_emitting, self._means, self._vars = False, None, None

        self.neigh, self.trans = [], []
        if loop:
            self.add_neigh(self, 1.)

        self.name = None

    def _shrink_transitions(self, free: float) -> None:
        self.trans = [t * (1. - free) for t in self.trans]

    def add_neigh(self, other_state, transition_probability) -> 'State':
        assert 0. <= transition_probability <= 1.
        self._shrink_transitions(transition_probability)
        self.neigh.append(other_state)
        self.trans.append(transition_probability)
        return self

    def emitting_logprobability(self, observation: FeatVec) -> float:
        return multivariate_normal.logpdf(observation, self._means, self._vars)

    def emit_observation(self) -> FeatVec:
        return multivariate_normal.rvs(self._means, self._vars)

    def update_distribution(self, data: List[FeatVec], past_importance: float = 0.) -> 'State':
        by_coordinate = zip(*data)
        for i, values in enumerate(by_coordinate):
            self._means[i] = (1. - past_importance) * mean(values) + past_importance * self._means[i]
            self._vars[i] = (1. - past_importance) * stdev(values) ** 2 + past_importance * self._vars[i]
        return self
