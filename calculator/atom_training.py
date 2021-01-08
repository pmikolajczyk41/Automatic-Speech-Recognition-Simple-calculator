import sys
from pathlib import Path

from calculator import ATOM_LENGTHS, ATOMS, BREAKS
from data import TRAIN_DIR
from data.provide import provide_mffcs
from hmm.model import PathModel, Model


def retrain_atoms(bw_iterations: int, new_version: str, base_version: str) -> None:
    for atom_name in ATOMS:
        sys.stderr.write(f'Retraining model for \'{atom_name}\'\n')
        data = provide_mffcs(TRAIN_DIR, atom_name)

        atom_model = Model.load(Path(f'trained/{base_version}{atom_name}.hmm'))
        # atom_model.render()

        atom_model.train_baum_welch(data, bw_iterations)
        # atom_model.render()

        sys.stderr.write(f'Training completed\n')

        filename = Path(f'trained/{new_version}{atom_name}.hmm')
        atom_model.save(filename)
        sys.stderr.write(f'Model saved as \'{filename}\'\n\n')


def train_atoms(viterbi_iterations: int, bw_iterations: int, version: str = '') -> None:
    for atom_name, emit_length in ATOM_LENGTHS.items():
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
    # train_atoms(0, 10, 'viterbi-')
    # retrain_atoms(10, 'bw-', 'viterbi-')
    for atom_name in BREAKS:
        atom_model = Model.load(Path(f'trained/bw-{atom_name}.hmm'))
        atom_model.render()
