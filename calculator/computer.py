from operator import add, sub, mul, truediv
from random import randint
from typing import List, Union, Callable

from calculator.recognizers import interpret


class Computer:
    _mapping = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'plus': add, 'minus': sub, 'times': mul, 'by': truediv,
    }

    _operator_display = {
        add: ' + ', sub: ' - ', mul: ' * ', truediv: ' / '
    }

    def compute(self, prediction: str) -> str:
        prediction = interpret(prediction)
        converted = self._convert(prediction)
        return self._compute(converted)

    def _convert(self, formula: str) -> List[Union[int, Callable]]:
        atoms = formula.split(' ')
        assert len(atoms) % 2
        converted = [self._mapping[a] for a in atoms]
        assert all(isinstance(x, int) for x in converted[0::2])
        assert all(x in self._operator_display.keys() for x in converted[1::2])
        return converted

    def _display_operand(self, operand) -> str:
        stripped = f'{operand:.2f}'.rstrip('0').rstrip('.')
        return stripped if stripped else '0'

    def _compute(self, converted: List[Union[int, Callable]]) -> str:
        result = float(converted[0])
        result_repr = self._display_operand(result)

        for operator, operand in zip(converted[1::2], converted[2::2]):
            result_repr += self._operator_display[operator] + self._display_operand(operand)
            try:
                result = operator(result, float(operand))
            except ZeroDivisionError:
                result = randint(0, 42)

        # -0.0 case
        if result == 0:
            result = 0

        return result_repr + f' = {self._display_operand(result)}'


if __name__ == '__main__':
    assert Computer().compute('one plus three') == '1 + 3 = 4'
    assert Computer().compute('nine by three') == '9 / 3 = 3'
    assert Computer().compute('one by four') == '1 / 4 = 0.25'
    assert Computer().compute('two times zero') == '2 * 0 = 0'
    assert Computer().compute('eight times nine') == '8 * 9 = 72'
    assert Computer().compute('three minus seven') == '3 - 7 = -4'

    assert Computer().compute('three minus seven times two') == '3 - 7 * 2 = -8'
    assert Computer().compute('one by six times eight') == '1 / 6 * 8 = 1.33'
    assert Computer().compute('nine plus nine plus nine minus nine') == '9 + 9 + 9 - 9 = 18'
    assert Computer().compute('zero times eight minus four times zero') == '0 * 8 - 4 * 0 = 0'

    assert '0 / 0 = ' in Computer().compute('zero by zero')
