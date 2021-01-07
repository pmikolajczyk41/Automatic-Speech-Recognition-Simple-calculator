import sys
from pathlib import Path

from data import TRAIN_DIR
from data.provide import provide_mffcs
from hmm.model import PathModel

ATOMS = {'zero': PathModel(4),
         # 'one'    : PathModel(3),
         # 'two'    : PathModel(3),
         # 'three'  : PathModel(3),
         # 'four'   : PathModel(3),
         # 'five'   : PathModel(4),
         # 'six'    : PathModel(4),
         # 'seven'  : PathModel(4),
         # 'eight'  : PathModel(5),
         # 'nine'   : PathModel(4),
         # 'plus'   : PathModel(4),
         # 'minus'  : PathModel(6),
         # 'times'  : PathModel(5),
         # 'by'     : PathModel(3),
         # 'silence': PathModel(2),
         }


def train_atoms(viterbi_iterations: int, bw_iterations: int) -> None:
    for atom_name, atom_model in ATOMS.items():
        sys.stderr.write(f'Training model for \'{atom_name}\'\n')
        data = provide_mffcs(TRAIN_DIR, atom_name)

        atom_model.train_uniform(data)
        atom_model.train_viterbi(data, viterbi_iterations)
        atom_model.train_baum_welch(data, bw_iterations)
        sys.stderr.write(f'Training completed\n')
        atom_model.save(Path(f'trained/{atom_name}.hmm'))
        sys.stderr.write(f'Model saved as \'trained/{atom_name}.hmm\'\n\n')

        atom_model.render()


if __name__ == '__main__':
    train_atoms(5, 2)
