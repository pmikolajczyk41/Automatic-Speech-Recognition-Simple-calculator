import sys
from pathlib import Path

from calculator import ATOMS
from data import TRAIN_DIR
from data.provide import provide_mffcs
from hmm.model import PathModel


def train_atoms(viterbi_iterations: int, bw_iterations: int, version: str = '') -> None:
    for atom_name, emit_length in ATOMS.items():
        sys.stderr.write(f'Training model for \'{atom_name}\'\n')
        data = provide_mffcs(TRAIN_DIR, atom_name)

        atom_model = PathModel(emit_length, label=atom_name)
        atom_model.train_uniform(data)

        atom_model.train_viterbi(data, viterbi_iterations)
        # atom_model.render()

        atom_model.train_baum_welch(data, bw_iterations)
        # atom_model.render()

        sys.stderr.write(f'Training completed\n')

        filename = Path(f'trained/{version}{atom_name}.hmm')
        atom_model.save(filename)
        sys.stderr.write(f'Model saved as \'{filename}\'\n\n')


if __name__ == '__main__':
    train_atoms(20, 0, 'viterbi-')
