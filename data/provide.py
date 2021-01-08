from pathlib import Path

import numpy as np


def provide_mffcs(dir: Path, keyword: str = '', with_title: bool = False):
    pattern = f'*{keyword}*.mfcc' if keyword != '' else '*.mfcc'
    if with_title:
        return [(str(mfcc_file.name)[:-5], np.loadtxt(str(mfcc_file)))
                for mfcc_file in dir.rglob(pattern)]
    return [np.loadtxt(str(mfcc_file))
            for mfcc_file in dir.rglob(pattern)]
