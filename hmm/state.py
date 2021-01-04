from typing import NamedTuple, List, Optional

from scipy.stats import multivariate_normal

from data import FeatVec


class Distribution(NamedTuple):
    means: List[float]
    variances: List[float]


default_distribution = Distribution(39 * [0.], 39 * [1.])


class State:
    def __init__(self, initial_distribution: Optional[Distribution] = None, loop: bool = False):
        if initial_distribution is not None:
            self.is_emitting, (self._means, self._vars) = True, initial_distribution
        else:
            self.is_emitting, self._means, self._vars = False, None, None

        self.neigh, self.trans = [], []
        if loop:
            self.add_neigh(self, 1.)

    def normalize_transitions(self) -> None:
        s = sum(self.trans)
        self.trans = [t / s for t in self.trans]

    def add_neigh(self, other_state, transition_probability) -> 'State':
        self.neigh.append(other_state)
        self.trans.append(transition_probability)
        self.normalize_transitions()
        return self

    def emitting_logprobability(self, observation: FeatVec) -> float:
        return multivariate_normal.logpdf(observation, self._means, self._vars)

    def emit_observation(self) -> FeatVec:
        return multivariate_normal.rvs(self._means, self._vars)


if __name__ == '__main__':
    s = State(initial_distribution=default_distribution, loop=False)
    print(s.emitting_logprobability(39 * [0.]))
    print()
    print(s.emit_observation())
