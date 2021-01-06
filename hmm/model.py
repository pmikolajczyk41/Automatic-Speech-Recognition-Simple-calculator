from collections import defaultdict
from itertools import chain
from random import randint
from typing import List, Iterable

import numpy as np
from graphviz import Digraph

from data import TRAIN_DIR, FeatVec
from data.provide import provide_mffcs
from hmm.baum_welch import baum_welch
from hmm.state import State, default_distribution
from hmm.viterbi import viterbi


class Model:
    @classmethod
    def Path(cls, emitting_length: int) -> 'Model':
        middle = [State(default_distribution, True) for _ in range(emitting_length)]
        states = [State()] + middle + [State()]

        states[0].add_neigh(states[1], 1.)
        for p, n in zip(states[1:], states[2:]):
            p.add_neigh(n, 0.5)

        return Model(states)

    def __init__(self, states: List[State]):
        self._states = states
        for id, state in enumerate(states):
            state.name = id

    def initial_state(self) -> State:
        return self._states[0]

    def target_state(self) -> State:
        return self._states[-1]

    def render(self) -> None:
        self._adj = [[(n.name, tp) for n, tp in zip(s.neigh, s.trans)]
                     for s in self._states]

        dot = Digraph(graph_attr={'rankdir': 'LR'}, node_attr={'shape': 'circle'})
        n = len(self._states)
        for s in self._states:
            if not s.is_emitting:
                dot.node(str(s.name), label='', width='0.2', style='filled')
        for u in range(n):
            for v, puv in self._adj[u]:
                dot.edge(str(u), str(v), label=f'{puv:.3f}'.rstrip('0').rstrip('.'))
        dot.render(f'model{randint(0, 10)}.gv', view=True)

    @staticmethod
    def _partition(data, nchunks: int):
        rwise_split = (np.array_split(row, nchunks) for row in data)
        transposed = zip(*rwise_split)
        return (list(chain.from_iterable(x)) for x in transposed)

    def train_uniform(self, data) -> None:
        # assuming that called on a result of Model.Path()
        nsamples = len(data)
        chunks = self._partition(data, len(self._states) - 2)
        for state, chunk in zip(self._states[1:-1], chunks):
            state.update_distribution(chunk)
            avg_loops = len(chunk) / float(nsamples)
            state.trans = [avg_loops / (avg_loops + 1), 1. / (avg_loops + 1)]

    def _assign_observations(self, mapping: defaultdict, obs_sequence: List[FeatVec], state_sequence: List) -> None:
        obs_id = 0
        for state_name in state_sequence:
            if self._states[state_name].is_emitting:
                mapping[state_name].append(obs_sequence[obs_id])
                obs_id += 1
        assert obs_id == len(obs_sequence)

    def _update_transitions(self, transitions: List[Iterable], past_importance: float) -> None:
        counters = {s.name: len(s.neigh) for s in self._states}
        nexts = {s.name: defaultdict(lambda: 1) for s in self._states}

        for old_t in transitions:
            for pred, succ in zip(old_t, old_t[1:]):
                counters[pred] += 1
                nexts[pred][succ] += 1
        for s in self._states:
            new_transitions = []
            for n, old_t in zip(s.neigh, s.trans):
                new_t = nexts[s.name][n.name] / counters[s.name]
                new_transitions.append((1. - past_importance) * new_t + past_importance * old_t)
            s.trans = new_transitions

    def train_viterbi(self, data, iterations: int, past_importance: float = 0.) -> None:
        for _ in range(iterations):
            transitions = []
            obs_mapping = defaultdict(list)

            for observation_sequence in data:
                token = viterbi(self.initial_state(), observation_sequence, self.target_state())
                print(token.log_probability, token.history)
                transitions.append(token.history)
                self._assign_observations(obs_mapping, observation_sequence, token.history)
            self._update_transitions(transitions, past_importance)
            for s in self._states:
                s.update_distribution(obs_mapping[s.name], past_importance)

    def _update_transitions_bw(self, gammas, ksis) -> None:
        for s in filter(lambda s: s.is_emitting, self._states):
            denominator = sum(gamma[s.name].sum() for gamma in gammas)
            for nid, n in enumerate(s.neigh):
                numerator = sum((ksi[s.name, n.name].sum()) for ksi in ksis)
                s.trans[nid] = numerator / denominator

    def _assign_observations_bw(self, state, gammas, data) -> Iterable[FeatVec]:
        if state.is_emitting:
            for observation_sequence, gamma in zip(data, gammas):
                probs = gamma[state.name] / gamma[state.name].sum()
                yield (probs[:, np.newaxis] * observation_sequence).sum(axis=0)

    def train_baum_welch(self, data, iterations: int) -> None:
        for _ in range(iterations):
            matrices = [baum_welch(self._states, observation_sequence) for observation_sequence in data]
            gammas, ksis = zip(*matrices)

            self._update_transitions_bw(gammas, ksis)
            for s in self._states:
                s.update_distribution(list(self._assign_observations_bw(s, gammas, data)))


if __name__ == '__main__':
    m = Model.Path(4)
    data = provide_mffcs(TRAIN_DIR)
    m.train_uniform(data)
    m.train_viterbi(data, 3)
    m.train_baum_welch(data, 5)
    m.render()
