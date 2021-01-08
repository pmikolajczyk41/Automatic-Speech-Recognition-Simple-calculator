import json
import sys
from collections import defaultdict
from itertools import chain
from pathlib import Path
from random import randint
from typing import List, Iterable, Set

import numpy as np
from graphviz import Digraph

from data import FeatVec
from hmm.baum_welch import baum_welch
from hmm.state import State, default_distribution
from hmm.viterbi import viterbi


class Model:
    def _type(self):
        raise NotImplementedError

    def initial_state(self) -> State:
        raise NotImplementedError

    def target_state(self) -> State:
        raise NotImplementedError

    def _get_all_states(self, current: State, accumulator: Set[State]) -> None:
        if current not in accumulator:
            accumulator.add(current)
            for n in current.neigh:
                self._get_all_states(n, accumulator)

    def render(self) -> None:
        states = set()
        self._get_all_states(self.initial_state(), states)

        dot = Digraph(graph_attr={'rankdir': 'LR'}, node_attr={'shape': 'circle'})
        for s in states:
            if not s.is_emitting:
                dot.node(str(id(s)), label=s.label, width='0.2', style='filled')
            else:
                dot.node(str(id(s)), label=s.label)
        for s in states:
            for v, puv in zip(s.neigh, s.trans):
                dot.edge(str(id(s)), str(id(v)), label=f'{puv:.3f}'.rstrip('0').rstrip('.'))
        dot.render(f'.vis/model{randint(0, 10)}.gv', view=True)

    def save(self, filename: Path) -> None:
        states = set()
        self._get_all_states(self.initial_state(), states)
        states = {s: i for i, s in enumerate(sorted(states, key=lambda s: s.rank))}

        data = {'type'         : self._type(),
                'initial_state': states[self.initial_state()],
                'target_state' : states[self.target_state()],
                'states'       : [s.serialize(states) for s in states.keys()]}

        filename.write_text(json.dumps(data))

    @staticmethod
    def load(filename: Path) -> 'Model':
        data = json.loads(filename.read_text())
        states = [State.deserialize(x) for x in data['states']]
        mapping = {i: states[i] for i in range(len(states))}
        for i, s in enumerate(states):
            s.recover_neighbourhood(data['states'][i], mapping)

        if data['type'] == 'path':
            return PathModel.from_states(states)
        if data['type'] == 'complex':
            return ComplexModel(states[data['initial_state']], states[data['target_state']])

    def predict(self, observation_sequence: Iterable[FeatVec]) -> str:
        winner = viterbi(self.initial_state(), observation_sequence, self.target_state())
        return winner.label


class PathModel(Model):
    def __init__(self, emitting_length: int, label: str = ''):
        middle = [State(default_distribution, rank=rank + 1) for rank in range(emitting_length)]
        self._states = [State(rank=0)] + middle + [State(label=label, rank=emitting_length + 1)]

        self._states[0].add_neigh(self._states[1], 1.)
        for p, n in zip(self._states[1:], self._states[2:]):
            p.add_neigh(n, 0.5)
            p.add_neigh(p, 0.5)

    @staticmethod
    def from_states(states: List[State]) -> 'PathModel':
        newborn = PathModel(0)
        newborn._states = states
        for i, s in enumerate(states):
            assert s.rank == i
        return newborn

    def _type(self):
        return 'path'

    def initial_state(self) -> State:
        return self._states[0]

    def target_state(self) -> State:
        return self._states[-1]

    @staticmethod
    def _partition(data, nchunks: int):
        rwise_split = (np.array_split(row, nchunks) for row in data)
        transposed = zip(*rwise_split)
        return (list(chain.from_iterable(x)) for x in transposed)

    def train_uniform(self, data) -> None:
        nsamples = len(data)
        chunks = self._partition(data, len(self._states) - 2)
        for state, chunk in zip(self._states[1:-1], chunks):
            state.update_distribution(chunk)
            avg_loops = len(chunk) / float(nsamples)
            state.trans = [avg_loops / (avg_loops + 1), 1. / (avg_loops + 1)]

    def _assign_observations(self, mapping: defaultdict, obs_sequence: List[FeatVec], state_sequence: List) -> None:
        obs_id = 0
        for state in state_sequence:
            if state.is_emitting:
                mapping[state].append(obs_sequence[obs_id])
                obs_id += 1
        assert obs_id == len(obs_sequence)

    def _update_transitions(self, transitions: List[Iterable]) -> None:
        counters = {s: len(s.neigh) for s in self._states}
        nexts = {s: defaultdict(lambda: 1) for s in self._states}

        for old_t in transitions:
            for pred, succ in zip(old_t, old_t[1:]):
                counters[pred] += 1
                nexts[pred][succ] += 1
        for s in self._states:
            new_transitions = []
            for n, old_t in zip(s.neigh, s.trans):
                new_t = nexts[s][n] / counters[s]
                new_transitions.append(new_t)
            s.trans = new_transitions

    def train_viterbi(self, data, iterations: int) -> None:
        last_transitions = None
        for it in range(iterations):
            sys.stderr.write(f'\rViterbi training: {100 * (it / iterations):.2f}%')

            transitions = []
            obs_mapping = defaultdict(list)

            for observation_sequence in data:
                token = viterbi(self.initial_state(), observation_sequence, self.target_state())
                transitions.append(token.history)
                self._assign_observations(obs_mapping, observation_sequence, token.history)

            if last_transitions == transitions:
                sys.stderr.write(f'\rViterbi training completed (only {it}/{iterations} iterations was needed)\n')
                return
            last_transitions = transitions

            self._update_transitions(transitions)
            for s, obs in obs_mapping.items():
                s.update_distribution(obs)

        sys.stderr.write(f'\rViterbi training completed\n')

    def _update_transitions_bw(self, gammas, ksis) -> None:
        for s in filter(lambda s: s.is_emitting, self._states):
            denominator = sum(gamma[s.rank].sum() for gamma in gammas)
            for nid, n in enumerate(s.neigh):
                numerator = sum((ksi[s.rank, n.rank].sum()) for ksi in ksis)
                s.trans[nid] = numerator / denominator

    def _assign_observations_bw(self, state, gammas, data) -> Iterable[FeatVec]:
        if state.is_emitting:
            for observation_sequence, gamma in zip(data, gammas):
                probs = gamma[state.rank] / gamma[state.rank].sum()
                yield (probs[:, np.newaxis] * observation_sequence).sum(axis=0)

    def train_baum_welch(self, data, iterations: int) -> None:
        for it in range(iterations):
            sys.stderr.write(f'\rBaumWelch training: {100 * (it / iterations):.2f}%')

            matrices = [baum_welch(self._states, observation_sequence) for observation_sequence in data]
            gammas, ksis = zip(*matrices)

            self._update_transitions_bw(gammas, ksis)
            for s in self._states:
                s.update_distribution(list(self._assign_observations_bw(s, gammas, data)))

        sys.stderr.write(f'\rBaumWelch training completed\n')


class ComplexModel(Model):
    def __init__(self, initial_state: State, target_state: State):
        assert (not target_state.is_emitting) and len(target_state.neigh) == 0

        self._initial_state = initial_state
        self._target_state = target_state

    def _type(self):
        return 'complex'

    def initial_state(self) -> State:
        return self._initial_state

    def target_state(self) -> State:
        return self._target_state

    def append(self, others: List[Model]) -> 'ComplexModel':
        new_target = State()
        for other in others:
            self._target_state.add_neigh(other.initial_state(), 1.0 / len(others))
            other.target_state().add_neigh(new_target, 1.0)
        self._target_state = new_target
        return self
