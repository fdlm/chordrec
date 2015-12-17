from __future__ import print_function
import sys
import os
import numpy as np
from data import write_chord_predictions, FPS
from nn.utils import Colors


PREDICTION_EXT = '.chords.txt'


def compute_labeling(network, agg_dataset, dest_dir, rnn,
                     extension='.chords.txt', out_onehot=True):
    """
    Computes and saves the labels for each datasource in an aggragated
    datasource
    :param network:     neural network
    :param agg_dataset: aggragated datasource.
    :param dest_dir:    where to store predicted chord labels
    :param rnn:         if the network is an rnn
    :param out_onehot:  if the output of a network is one-hot encoded
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
        ds = agg_dataset.get_datasource(ds_idx)

        # skip targets
        data, _ = ds[:]

        if rnn:
            data = data[np.newaxis, :]
            mask = np.ones(data.shape[:-1], dtype=np.float32)

            pred = network.process(data, mask)[0]
        else:
            pred = network.process(data)

        if out_onehot:
            # one hot encoding to class id
            pred = pred.argmax(axis=1)

        pred_file = os.path.join(dest_dir, ds.name + extension)
        write_chord_predictions(filename=pred_file, predictions=pred, fps=FPS)
        pred_files.append(pred_file)

    return pred_files


def compute_average_scores(annotation_files, prediction_files):
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
            (song_length,
             mir_eval.chord.evaluate(ann_int, ann_lab, pred_int, pred_lab))
        )

    # initialise the average score with all metrics and values 0.
    avg_score = {metric: 0. for metric in scores[0][1]}

    for length, score in scores:
        weight = length / total_length
        for metric in score:
            avg_score[metric] += weight * score[metric]

    return avg_score


def print_scores(scores):
    for name, val in scores.iteritems():
        label = '\t{}:'.format(name).ljust(16)
        print(label + '{:.3f}'.format(val))
