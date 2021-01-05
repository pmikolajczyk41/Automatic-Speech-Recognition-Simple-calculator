from itertools import chain
from typing import List

import numpy as np
from graphviz import Digraph

from data import TRAIN_DIR
from data.provide import provide_mffcs
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
        dot.render('model.gv', view=True)

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


if __name__ == '__main__':
    m = Model.Path(5)
    data = provide_mffcs(TRAIN_DIR)
    m.train_uniform(data)
    winner = viterbi(m.initial_state(), data[0], m.target_state())
    print(winner.log_probability, winner.history)
