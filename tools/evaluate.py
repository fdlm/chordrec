import os
import fnmatch
from docopt import docopt

import dmgr

from chordrec import test


USAGE = """
evaluate.py

Usage:
    evaluate.py [-i IND_RES_FILE] [-o TOT_RES_FILE] FILES...

Arguments:
    FILES  annotaion or prediction files

Options:
    -i IND_RES_FILE  file where to store individual results
    -o TOT_RES_FILE  file where to store total results
"""


def main():
    args = docopt(USAGE)

    ann_files = fnmatch.filter(args['FILES'], '*.chords')

    pred_files = dmgr.files.match_files(
        ann_files, '.chords',
        args['FILES'], '.chords.txt'
    )

    test.print_scores(test.compute_average_scores(ann_files, pred_files))


if __name__ == "__main__":
    main()
