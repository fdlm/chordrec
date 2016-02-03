import os
import string

import numpy as np

import dmgr

DATA_DIR = 'data'
CACHE_DIR = 'feature_cache'
SRC_EXT = '.flac'
GT_EXT = '.chords'


def chords_maj_min(target_file, num_frames, fps):
    """
    Creates one-hot encodings from a chord annotation file. Right now,
    chords are mapped to major/minor, resulting in 24 chord classes and one
    'no chord' class.

    :param target_file: file containing chord annotations
    :param num_frames:  number of frames in the audio file
    :param fps:         frames per second
    :return:            one-hot ground truth per frame
    """
    # first, create chord/class mapping. root note 'A' has id 0, increasing
    # with each semitone. we have duplicate mappings for flat and sharp notes,
    # just to be sure.
    natural = zip(string.uppercase[:7], [0, 2, 3, 5, 7, 8, 10])
    sharp = map(lambda v: (v[0] + '#', (v[1] + 1) % 12), natural)
    flat = map(lambda v: (v[0] + 'b', (v[1] - 1) % 12), natural)

    # 'no chord' is coded as 'N'. The class ID of 'N' is 24, after all major
    # and minor chords. Sometimes there is also an 'X' annotation, meaning
    # that the chord cannot be properly determined on beat-lebel (too much
    # going on in the audio). We will treat this also as 'no chord'
    root_note_map = dict(natural + sharp + flat + [('N', 24), ('X', 24)])

    # then, we load the annotations, map the chords to class ids, and finally
    # map class ids to a one-hot encoding. first, map the root notes.
    ann = np.loadtxt(target_file, dtype=str)
    chord_names = ann[:, -1]
    chord_root_notes = [c.split(':')[0].split('/')[0] for c in chord_names]
    chord_root_note_ids = np.array([root_note_map[crn]
                                    for crn in chord_root_notes])

    # then, map the chords to major and minor. we assume chords with a minor
    # third as first interval are considered minor chords,
    # the rest are major chords, following MIREX, as stated in
    # Taemin Cho, Juan Bello: "On the relative importance of Individual
    # Components of Chord Recognition Systems"

    chord_type = [c.split(':')[1] if ':' in c else '' for c in chord_names]

    # we will shift the class ids for all minor notes by 12 (num major chords)
    chord_type_shift = np.array(
        map(lambda x: 12 if 'min' in x or 'dim' in x else 0, chord_type)
    )

    # now we can compute the final chord class id
    chord_class_id = chord_root_note_ids + chord_type_shift

    n_chords = len(chord_class_id)
    # 25 classes - 12 major, 12 minor, one no chord
    # we will add a dummy 'NO CHORD' at the end and at the beginning,
    # because some annotations miss it, are not exactly aligned at the end
    # or do not start at the beginning of an audio file
    one_hot = np.zeros((n_chords + 2, 25), dtype=np.int32)
    one_hot[np.arange(n_chords) + 1, chord_class_id] = 1
    # these are the dummy 'NO CHORD' annotations
    one_hot[0, 24] = 1
    one_hot[-1, 24] = 1

    # make sure everything is in its place
    assert (one_hot.argmax(axis=1)[1:-1] == chord_class_id).all()
    assert (one_hot.sum(axis=1) == 1).all()

    # Now, we create the time stamps. if no explicit end times are given,
    # we take the start time of the next chord as end time for the current.
    start_ann = ann[:, 0].astype(np.float)
    end_ann = (ann[:, 1].astype(np.float) if ann.shape[1] > 2 else
               np.hstack((start_ann[1:], [np.inf])))

    # add the times for the dummy events
    start = np.hstack(([-np.inf], start_ann, end_ann[-1]))
    end = np.hstack((start_ann[0], end_ann, [np.inf]))

    # Finally, we create the one-hot encoding per frame!
    frame_times = np.arange(num_frames, dtype=np.float) / fps

    # IMPORTANT: round everything to milliseconds to prevent errors caused
    # by floating point hell. Ideally, we would round everything to
    # possible *frame times*, but it is easier this way.
    start = np.round(start, decimals=3)
    end = np.round(end, decimals=3)
    frame_times = np.round(frame_times, decimals=3)

    target_per_frame = ((start <= frame_times[:, np.newaxis]) &
                        (frame_times[:, np.newaxis] < end))

    # make sure each frame is assigned to only one target vector
    assert (target_per_frame.sum(axis=1) == 1).all()

    # create the one hot vectors per frame
    return one_hot[np.nonzero(target_per_frame)[1]].astype(np.float32)


def predictions_to_chord_label(predictions, fps):
    natural = zip([0, 2, 3, 5, 7, 8, 10], string.uppercase[:7])
    sharp = map(lambda v: ((v[0] + 1) % 12, v[1] + '#'), natural)

    semitone_to_label = dict(sharp + natural)

    def pred_to_cl(pred):
        if pred == 24:
            return 'N'
        return '{}:{}'.format(semitone_to_label[pred % 12],
                              'maj' if pred < 12 else 'min')

    spf = 1. / fps
    labels = [(i * spf, pred_to_cl(p)) for i, p in enumerate(predictions)]

    # join same consequtive predictions
    prev_label = (None, None)
    uniq_labels = []

    for label in labels:
        if label[1] != prev_label[1]:
            uniq_labels.append(label)
            prev_label = label

    # end time of last label is one frame duration after
    # the last prediction time
    start_times, chord_labels = zip(*uniq_labels)
    end_times = start_times[1:] + (labels[-1][0] + spf,)

    return zip(start_times, end_times, chord_labels)


def write_chord_predictions(filename, predictions, fps):
    with open(filename, 'w') as f:
        f.writelines(['{:.3f}\t{:.3f}\t{}\n'.format(*p)
                      for p in predictions_to_chord_label(predictions, fps)])


def combine_files(*args):
    """
    Combines file dictionaries as returned by the methods of Dataset.
    :param args: file dictionaries
    :return:     combined file dictionaries
    """

    combined = {'train': {'feat': [],
                          'targ': []},
                'val': {'feat': [],
                        'targ': []},
                'test': {'feat': [],
                         'targ': []}}

    for fs in args:
        for s in combined:
            for t in combined[s]:
                combined[s][t] += fs[s][t]

    return combined


DATASET_DEFS = {
    'beatles': {
        'data_dir': 'beatles',
        'split_filename': '8-fold_cv_album_distributed_{}.fold'
    },
    'queen': {
        'data_dir': 'queen',
        'split_filename': '8-fold_cv_random_{}.fold'
    },
    'zweieck': {
        'data_dir': 'zweieck',
        'split_filename': '8-fold_cv_random_{}.fold'
    },
    'robbie_williams': {
        'data_dir': 'robbie_williams',
        'split_filename': '8-fold_cv_random_{}.fold'
    },
    'billboard': {
        'data_dir': os.path.join('mcgill-billboard', 'unique'),
        'split_filename': '8-fold_cv_random_{}.fold'
    }
}


def load_dataset(name, data_dir, feature_cache_dir,
                 compute_features, compute_targets):

    assert name in DATASET_DEFS.keys(), 'Unknown dataset {}'.format(name)

    data_dir = os.path.join(data_dir, DATASET_DEFS[name]['data_dir'])
    split_filename = os.path.join(data_dir, 'splits',
                                  DATASET_DEFS[name]['split_filename'])

    return dmgr.Dataset(
        data_dir,
        os.path.join(feature_cache_dir, name),
        [split_filename.format(f) for f in range(8)],
        source_ext=SRC_EXT,
        gt_ext=GT_EXT,
        compute_features=compute_features,
        compute_targets=compute_targets,
    )


def create_preprocessors(preproc_def):
    preprocessors = []
    for pp_name, pp_params in preproc_def:
        preprocessors.append(getattr(dmgr.preprocessing, pp_name)(**pp_params))
    return preprocessors


def create_datasources(dataset_names, preprocessors,
                       compute_features, compute_targets, context_size,
                       data_dir=DATA_DIR, feature_cache_dir=CACHE_DIR,
                       test_fold=0, val_fold=None,
                       **kwargs):

    val_fold = val_fold or test_fold - 1
    preprocessors = create_preprocessors(preprocessors)

    if context_size > 0:
        data_source_type = dmgr.datasources.ContextDataSource
        kwargs['context_size'] = context_size
    else:
        data_source_type = dmgr.datasources.DataSource

    # load all datasets
    datasets = [load_dataset(name, data_dir, feature_cache_dir,
                             compute_features, compute_targets)
                for name in dataset_names]

    # uses fold 0 for validation, fold 1 for test, rest for training
    train, val, test = dmgr.datasources.get_datasources(
        combine_files(*[ds.get_fold_split(val_fold, test_fold)
                        for ds in datasets]),
        preprocessors=preprocessors, data_source_type=data_source_type,
        **kwargs
    )

    return train, val, test, sum((ds.gt_files for ds in datasets), [])


def add_sacred_config(ex):
    ex.add_config(
        datasource=dict(
            datasets=['beatles', 'queen', 'zweieck'],
            context_size=0,
            preprocessors=[],
            test_fold=0,
            val_fold=None
        )
    )
