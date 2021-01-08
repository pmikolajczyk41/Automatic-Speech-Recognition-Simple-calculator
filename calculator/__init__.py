ATOM_LENGTHS = {
    'zero'   : 6,
    'one'    : 7,
    'two'    : 6,
    'three'  : 5,
    'four'   : 5,
    'five'   : 5,
    'six'    : 7,
    'seven'  : 7,
    'eight'  : 6,
    'nine'   : 6,
    'plus'   : 6,
    'minus'  : 6,
    'times'  : 6,
    'by'     : 5,
    'silence': 2,
}

DIGITS = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', }
OPERATORS = {'plus', 'minus', 'times', 'by'}
BREAKS = {'silence'}

ATOMS = DIGITS | OPERATORS | BREAKS
