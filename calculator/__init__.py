ATOM_LENGTHS = {
    'zero'   : 18,
    'one'    : 9,
    'two'    : 6,
    'three'  : 9,
    'four'   : 9,
    'five'   : 12,
    'six'    : 12,
    'seven'  : 12,
    'eight'  : 9,
    'nine'   : 12,
    'plus'   : 12,
    'minus'  : 18,
    'times'  : 15,
    'by'     : 9,
    'silence': 2,
}

DIGITS = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', }
OPERATORS = {'plus', 'minus', 'times', 'by'}
BREAKS = {'silence'}

ATOMS = DIGITS | OPERATORS | BREAKS
