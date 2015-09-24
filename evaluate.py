import os
from glob import glob
from docopt import docopt

import dmgr

import test


USAGE = """
evaluate.py

Usage:
    evaluate.py ANN_DIR PRED_DIR

Arguments:
    ANN_DIR  directory containing annotations
    PRED_DIR  directory containing predictions
"""


def main():
    args = docopt(USAGE)

    pred_files = glob(os.path.join(args['PRED_DIR'], '*.chords.txt'))
    ann_files = dmgr.files.match_files(
        pred_files,
        glob(os.path.join(args['ANN_DIR'], '*.chords')),
        '.chords.txt',
        '.chords'
    )

    test.print_scores(test.compute_average_scores(ann_files, pred_files))


if __name__ == "__main__":
    main()