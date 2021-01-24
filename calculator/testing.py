import multiprocessing
import sys
from itertools import product
from pathlib import Path
from random import shuffle
from typing import List, Tuple, Iterable

from calculator import ATOMS
from calculator.recognizers import create_atom_recognizer, interpret
from calculator.recognizers import create_single_operation_recognizer
from data import TEST_DIR, FeatVec
from data.provide import provide_mffcs
from hmm.model import Model


def predict_single(model: Model, labeled_sample: Tuple[str, Iterable[FeatVec]]) -> str:
    return interpret(model.predict(labeled_sample[1], margin_size=7000.))


def test(data, model, parallel: bool = False):
    shuffle(data)
    history = []

    if parallel:
        with multiprocessing.Pool(processes=3) as pool:
            for (label, _), pred in zip(data, pool.starmap(predict_single, product([model], data))):
                history.append((label, pred))
    else:
        for it, (label, sample) in enumerate(data):
            sys.stderr.write(f'\rTesting in progress: {100 * (it / len(data)):.2f}%')
            history.append((label, predict_single(model, (label, sample))))

    ok = sum((label == pred) for label, pred in history)
    sys.stderr.write(f'\rTesting completed: accuracy {100 * (ok / len(data)):.2f}%\nSummary:\n')

    for label, pred in history:
        sys.stderr.write(f'\tlabel: <{label}>\t\tprediction: <{pred}>\n')


def test_atoms(data_dir: Path, atom_names: List[str] = ATOMS, version: str = '') -> None:
    data = [(an, sample) for an in atom_names for sample in provide_mffcs(data_dir, an)]
    model = create_atom_recognizer(atom_names, version)
    test(data, model)


def test_operation(data_dir, version: str = '') -> None:
    data = provide_mffcs(data_dir, with_title=True)
    model = create_single_operation_recognizer(version)
    test(data, model)


if __name__ == '__main__':
    # data_dir = TEST_DIR / 'atoms' / 'default-speaker'
    # test_atoms(data_dir, version='viterbi-')
    # test_atoms(data_dir, version='bw-')
    # test_atoms(data_dir, version='bw50-')

    data_dir = TEST_DIR / 'operations' / 'complex-operation' / 'default-speaker'
    # test_operation(data_dir, version='viterbi-')
    # test_operation(data_dir, version='bw-')
    test_operation(data_dir, version='bw50-')
