from pathlib import Path
from typing import List

from calculator import ATOMS, DIGITS, OPERATORS, BREAKS
from hmm.model import ComplexModel, Model
from hmm.state import State


def interpret(pred: str) -> str:
    return pred.replace('silence', ' ').strip()


def create_atom_recognizer(atom_names: List[str] = ATOMS, version: str = '') -> Model:
    root = State()
    atom_models = [Model.load(Path(f'trained/{version}{an}.hmm')) for an in atom_names]
    if len(atom_models) == 1:
        return atom_models[0]
    return ComplexModel(root, root).append(atom_models)


def create_single_operation_recognizer(version: str = '') -> ComplexModel:
    first_digit_recognizer = create_atom_recognizer(DIGITS, version=version)
    second_digit_recognizer = create_atom_recognizer(DIGITS, version=version)
    operator_recognizer = create_atom_recognizer(OPERATORS, version=version)
    first_silence_recognizer = create_atom_recognizer(BREAKS, version=version)
    second_silence_recognizer = create_atom_recognizer(BREAKS, version=version)

    root = State()
    return (ComplexModel(root, root)
            .append([first_digit_recognizer])
            .append([first_silence_recognizer])
            .append([operator_recognizer])
            .append([second_silence_recognizer])
            .append([second_digit_recognizer]))
