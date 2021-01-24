from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import delta
from python_speech_features import mfcc

from data import FeatVec, TEST_DIR


def wav2mfcc(rate, sig) -> FeatVec:
    assert rate == 16000
    mfcc_feat = mfcc(sig, rate)
    delta_feat = delta(mfcc_feat, 2)
    acc_feat = delta(delta_feat, 2)
    return np.hstack((mfcc_feat, delta_feat, acc_feat))


def convert_file(wavefile: Path) -> None:
    rate, sig = wav.read(wavefile)
    features = wav2mfcc(rate, sig)
    np.savetxt(wavefile.with_suffix('.mfcc'), features)


def convert_files_within_dir(directory: Path) -> None:
    for wavefile in directory.rglob('*.wav'):
        convert_file(wavefile)


if __name__ == '__main__':
    convert_files_within_dir(TEST_DIR)
