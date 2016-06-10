from __future__ import print_function
import sys
import os
import numpy as np

from dmgr.iterators import iterate_batches
from nn.utils import Colors


PREDICTION_EXT = '.chords.txt'


def compute_labeling(process_fn, target, agg_dataset, dest_dir, use_mask,
                     batch_size=None, extension='.chords.txt'):
    """
    Computes and saves the labels for each datasource in an aggragated
    datasource
    :param process_fn:  theano function that gives the nn's output
    :param target:      target computer
    :param agg_dataset: aggragated datasource.
    :param dest_dir:    where to store predicted chord labels
    :param use_mask:    if the network is an rnn
    :param batch_size:  Batch size if each datasource is to be processed batch-wise
    :param extension:   file extension of the resulting files
    :return:            list of files containing the predictions
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    else:
        if not os.path.isdir(dest_dir):
            print(Colors.red('Destination path exists but is not a directory!'),
                  file=sys.stderr)
            return

    pred_files = []

    for ds_idx in range(agg_dataset.n_datasources):
        ds = agg_dataset.datasource(ds_idx)

        pred = []
        for data, _ in iterate_batches(ds, batch_size or ds.n_data,
                                       randomise=False, expand=False):
            if use_mask:
                data = data[np.newaxis, :]
                mask = np.ones(data.shape[:2], dtype=np.float32)

                p = process_fn(data, mask)[0]
            else:
                p = process_fn(data)

            pred.append(p.argmax(axis=1))

        pred = np.concatenate(pred)

        pred_file = os.path.join(dest_dir, ds.name + extension)
        target.write_chord_predictions(pred_file, pred)
        pred_files.append(pred_file)

    return pred_files


def compute_scores(annotation_files, prediction_files):
    assert len(annotation_files) == len(prediction_files)
    assert len(annotation_files) > 0
    import mir_eval

    scores = []
    total_length = 0.

    for af, pf in zip(annotation_files, prediction_files):
        ann_int, ann_lab = mir_eval.io.load_labeled_intervals(af)
        pred_int, pred_lab = mir_eval.io.load_labeled_intervals(pf)

        # we assume that the end-time of the last annotated label is the
        # length of the song
        song_length = ann_int[-1][1]
        total_length += song_length

        scores.append(
            (pf, song_length,
             mir_eval.chord.evaluate(ann_int, ann_lab, pred_int, pred_lab))
        )

    return scores, total_length


def average_scores(scores, total_length):
    # initialise the average score with all metrics and values 0.
    avg_score = {metric: 0. for metric in scores[0][-1]}

    for _, length, score in scores:
        weight = length / total_length
        for metric in score:
            avg_score[metric] += float(weight * score[metric])

    return avg_score


def compute_average_scores(annotation_files, prediction_files):
    # first, compute all individual scores
    scores, total_length = compute_scores(annotation_files, prediction_files)
    return average_scores(scores, total_length)


def print_scores(scores):
    for name, val in scores.iteritems():
        label = '\t{}:'.format(name).ljust(16)
        print(label + '{:.3f}'.format(val))
