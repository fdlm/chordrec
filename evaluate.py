import os
from glob import glob
from docopt import docopt

import dmgr

import test


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

    ann_files = dmgr.files.match_files_single(
        args['FILES'], '.chords.txt', '.chords')

    pred_files = dmgr.files.match_files(
        ann_files,
        args['FILES'],
        '.chords',
        '.chords.txt'
    )

    scores, total_length = test.compute_scores(ann_files, pred_files)

    if args['-i']:
        header = '# song, length, ' + ', '.join(scores[0][-1].keys())

    test.print_scores(test.compute_average_scores(ann_files, pred_files))


if __name__ == "__main__":
    main()
