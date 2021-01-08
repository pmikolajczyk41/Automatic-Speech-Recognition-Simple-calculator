import sys
from pathlib import Path
from random import shuffle
from typing import List

from calculator import ATOMS
from data import TRAIN_DIR
from data.provide import provide_mffcs
from hmm.model import ComplexModel, Model
from hmm.state import State


def create_digit_recognizer(atom_names: List[str] = ATOMS.keys()) -> ComplexModel:
    root = State()
    model = ComplexModel(root, root)
    model.append([Model.load(Path(f'trained/{an}.hmm')) for an in atom_names])
    return model


def test_atoms(atom_names: List[str] = ATOMS.keys()) -> None:
    data = [(sample, an) for an in atom_names for sample in provide_mffcs(TRAIN_DIR, an)]
    shuffle(data)

    model = create_digit_recognizer(atom_names)

    nsamples, ok = len(data), 0
    for it, (sample, label) in enumerate(data):
        sys.stderr.write(f'\rTesting in progress: {100 * (it / nsamples):.2f}%')
        pred = model.predict(sample)
        if pred == label:
            ok += 1
    sys.stderr.write(f'\rTesting completed: accuracy {100 * (ok / nsamples):.2f}%')


if __name__ == '__main__':
    test_atoms()
