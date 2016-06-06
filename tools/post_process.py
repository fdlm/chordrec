import numpy as np
import dmgr
import test
import os
import shutil
import fnmatch
import scipy.stats
from targets import ChordsMajMin
from docopt import docopt
from experiment import TempDir
from itertools import tee, izip

USAGE = """
Post-Processes chord prediction files.

Usage:
    post_process.py [options] <files>...

Options:
    --fps=<fps>  work with this number of frames per second [default: 10]
    --win_length=<win_length>  length in seconds of the post-processing filter
                               [default: 1.0]
    --beats  use beat-based majority vote
    --out_dir=<out_dir>  where to put the post-processed results
"""


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def majority_vote(targets, win_size):
    context_size = (win_size - 1) / 2
    t_wins = dmgr.datasources.segment_axis(targets, frame_size=win_size)
    middle = scipy.stats.mode(t_wins, axis=1)[0][:, 0]
    start = np.hstack([scipy.stats.mode(targets[:i + 1])[0]
                       for i in range(context_size)])
    end = np.hstack([scipy.stats.mode(targets[i:])[0]
                     for i in range(-context_size, 0)])
    return np.hstack((start, middle, end))


def majority_vote_beats(targets, beats):
    if len(beats) == 0:
        return targets
    pp_targets = np.zeros_like(targets)
    beats = np.concatenate(([0], beats, [None]))
    for start, end in pairwise(beats):
        pp_targets[start:end] = scipy.stats.mode(targets[start:end])[0]
    return pp_targets


def main():
    args = docopt(USAGE)

    fps = float(args['--fps'])
    win_size = int(float(args['--win_length']) * fps)
    if win_size % 2 == 0:
        win_size += 1

    out_dir = args['--out_dir']

    files = args['<files>']
    ann_files = fnmatch.filter(files, '*.chords')
    pred_files = dmgr.files.match_files(ann_files, '.chords',
                                        files, '.chords.txt')

    if args['--beats']:
        beat_files = dmgr.files.match_files(ann_files, files,
                                            '.chords', '.beats')
    else:
        beat_files = None

    pre_filter_scores = test.compute_average_scores(ann_files, pred_files)
    print "Pre-Filter scores:"
    test.print_scores(pre_filter_scores)

    with TempDir() as tmpdir:
        target = ChordsMajMin(fps)
        pp_pred_files = []
        for i, pf in enumerate(pred_files):
            name = os.path.basename(pf)
            targets = target(pf).argmax(axis=1)

            if not args['--beats']:
                pp_targets = majority_vote(targets, win_size)
            else:
                beats = np.loadtxt(beat_files[i], usecols=[0]) * fps
                pp_targets = majority_vote_beats(targets, beats)

            target.write_chord_predictions(
                os.path.join(tmpdir, name),
                pp_targets
            )
            pp_pred_files.append(os.path.join(tmpdir, name))

        post_filter_scores = test.compute_average_scores(ann_files,
                                                         pp_pred_files)
        print "Post-Filter scores:"
        test.print_scores(post_filter_scores)

        if out_dir is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            for f in pp_pred_files:
                shutil.move(f, os.path.join(out_dir, os.path.basename(f)))


if __name__ == '__main__':
    main()
