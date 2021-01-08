import sys
from pathlib import Path
from random import shuffle
from typing import List

from calculator import ATOMS
from data import TEST_DIR
from data.provide import provide_mffcs
from hmm.model import ComplexModel, Model
from hmm.state import State


def create_digit_recognizer(atom_names: List[str] = ATOMS.keys(), version: str = '') -> ComplexModel:
    root = State()
    model = ComplexModel(root, root)
    model.append([Model.load(Path(f'trained/{version}{an}.hmm')) for an in atom_names])
    return model


def test_atoms(data_dir: Path, atom_names: List[str] = ATOMS.keys(), version: str = '') -> None:
    data = [(sample, an) for an in atom_names for sample in provide_mffcs(data_dir, an)]
    shuffle(data)

    model = create_digit_recognizer(atom_names, version)

    nsamples, ok = len(data), 0
    for it, (sample, label) in enumerate(data):
        sys.stderr.write(f'\rTesting in progress: {100 * (it / nsamples):.2f}%')
        pred = model.predict(sample)
        if pred == label:
            ok += 1
    sys.stderr.write(f'\rTesting completed: accuracy {100 * (ok / nsamples):.2f}%')


if __name__ == '__main__':
    data_dir = TEST_DIR / 'atoms' / 'speaker-1'
    test_atoms(data_dir, version='viterbi-')
