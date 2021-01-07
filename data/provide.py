from pathlib import Path

import numpy as np


def provide_mffcs(dir: Path, keyword: str = ''):
    return [np.loadtxt(str(mfcc_file))
            for mfcc_file in dir.rglob(f'*{keyword}*.mfcc')]
