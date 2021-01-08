from copy import deepcopy
from statistics import mean, stdev
from typing import NamedTuple, List, Optional

import numpy as np
from scipy.stats import multivariate_normal

from data import FeatVec


class Distribution(NamedTuple):
    means: List[float]
    variances: List[float]


DIMENSIONALITY = 39
default_distribution = Distribution(DIMENSIONALITY * [0.], DIMENSIONALITY * [1.])

NULL_OBSERVATION = np.full(DIMENSIONALITY, np.nan)


class State:
    def __init__(self, initial_distribution: Optional[Distribution] = None, label: str = '', rank: int = 0):
        if initial_distribution is not None and None not in initial_distribution:
            self.is_emitting, (self._means, self._vars) = True, deepcopy(initial_distribution)
        else:
            self.is_emitting, self._means, self._vars = False, None, None

        self.neigh, self.trans = [], []
        self.label = label
        self.rank = rank

    def add_neigh(self, other_state, transition_probability) -> 'State':
        self.neigh.append(other_state)
        self.trans.append(transition_probability)
        return self

    def normalize_transitions(self) -> None:
        s = sum(self.trans)
        self.trans = [t / s for t in self.trans]

    def emitting_logprobability(self, observation: FeatVec) -> float:
        if (np.isnan(observation).any()):
            return -np.inf if self.is_emitting else 0.
        if not self.is_emitting:
            return -np.inf
        return multivariate_normal.logpdf(observation, self._means, self._vars)

    def emit_observation(self) -> Optional[FeatVec]:
        if self.is_emitting:
            return multivariate_normal.rvs(self._means, self._vars)
        return NULL_OBSERVATION

    def update_distribution(self, data: List[FeatVec]) -> 'State':
        if self.is_emitting:
            by_coordinate = zip(*data)
            for i, values in enumerate(by_coordinate):
                self._means[i] = mean(values)
                self._vars[i] = stdev(values) ** 2
        return self

    def serialize(self, state_mapping: dict) -> dict:
        return {'distribution': Distribution(self._means, self._vars),
                'neigh'       : [state_mapping[n] for n in self.neigh],
                'trans'       : self.trans,
                'label'       : self.label,
                'rank'        : self.rank}

    @staticmethod
    def deserialize(data: dict) -> 'State':
        return State(data['distribution'], data['label'], data['rank'])

    def recover_neighbourhood(self, data: dict, state_mapping: dict) -> 'State':
        self.trans = data['trans']
        self.neigh = [state_mapping[s] for s in data['neigh']]
        return self
