import os
import string

import numpy as np

import madmom as mm
import dmgr

FPS = 10
DATA_DIR = 'data'
CACHE_DIR = 'feature_cache'
SRC_EXT = '.flac'
GT_EXT = '.chords'


def compute_targets(target_file, num_frames, fps):
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


def compute_features(audio_file, fps):
    """
    This function just computes the features for an audio file
    :param audio_file: audio file to compute the features for
    :param fps: frames per second
    :return: features as numpy array (or similar)
    """

    # do not resample because ffmpeg/avconv creates terrible sampling
    # artifacts
    specs = [
        mm.audio.spectrogram.LogarithmicFilteredSpectrogram(
            audio_file, num_channels=1, sample_rate=44100, fps=fps,
            frame_size=ffts, num_bands=24, fmax=5500,
            unique_filters=False)
        for ffts in [8192 * 2]  # corresponds to ~0.37 seconds
    ]

    return np.hstack(specs).astype(np.float32)


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


def load_beatles_dataset(data_dir=DATA_DIR, feature_cache_dir=CACHE_DIR):
    return dmgr.Dataset(
        os.path.join(data_dir, 'beatles'),
        os.path.join(feature_cache_dir, 'beatles'),
        [os.path.join(data_dir, 'beatles', 'splits',
                      '8-fold_cv_album_distributed_{}.fold'.format(f))
         for f in range(8)],
        source_ext=SRC_EXT,
        gt_ext=GT_EXT,
        compute_features=compute_features,
        compute_targets=compute_targets,
        fps=FPS
    )


def load_queen_dataset(data_dir=DATA_DIR, feature_cache_dir=CACHE_DIR):
    return dmgr.Dataset(
            os.path.join(data_dir, 'queen'),
            os.path.join(feature_cache_dir, 'queen'),
            [os.path.join(data_dir, 'queen', 'splits',
                          '8-fold_cv_random_{}.fold'.format(f))
             for f in range(8)],
            source_ext=SRC_EXT,
            gt_ext=GT_EXT,
            compute_features=compute_features,
            compute_targets=compute_targets,
            fps=FPS
    )


def load_zweieck_dataset(data_dir=DATA_DIR, feature_cache_dir=CACHE_DIR):
    return dmgr.Dataset(
            os.path.join(data_dir, 'zweieck'),
            os.path.join(feature_cache_dir, 'zweieck'),
            [os.path.join(data_dir, 'zweieck', 'splits',
                          '8-fold_cv_random_{}.fold'.format(f))
             for f in range(8)],
            source_ext=SRC_EXT,
            gt_ext=GT_EXT,
            compute_features=compute_features,
            compute_targets=compute_targets,
            fps=FPS
    )


def load_robbie_dataset(data_dir=DATA_DIR, feature_cache_dir=CACHE_DIR):
    return dmgr.Dataset(
        os.path.join(data_dir, 'robbie_williams'),
        os.path.join(feature_cache_dir, 'robbie_williams'),
        [os.path.join(data_dir, 'robbie_williams', 'splits',
                      '8-fold_cv_random_{}.fold'.format(f))
         for f in range(8)],
        source_ext=SRC_EXT,
        gt_ext=GT_EXT,
        compute_features=compute_features,
        compute_targets=compute_targets,
        fps=FPS
    )


def load_billboard_dataset(data_dir=DATA_DIR, feature_cache_dir=CACHE_DIR):
    return dmgr.Dataset(
        os.path.join(data_dir, 'mcgill-billboard', 'unique'),
        os.path.join(feature_cache_dir, 'mcgill-billboard',),
        [os.path.join(data_dir, 'mcgill-billboard', 'splits',
                      '8-fold_cv_random_{}.fold'.format(f))
         for f in range(8)],
        source_ext=SRC_EXT,
        gt_ext=GT_EXT,
        compute_features=compute_features,
        compute_targets=compute_targets,
        fps=FPS
    )


def load_datasets(beatles=True, queen=True, zweieck=True, robbie=False,
                  billboard=False, data_dir=DATA_DIR,
                  feature_cache_dir=CACHE_DIR, **kwargs):

    assert beatles or queen or zweieck or robbie or billboard, \
        "Load at least one dataset"

    datasets = []

    if beatles:
        datasets.append(load_beatles_dataset(data_dir, feature_cache_dir))
    if queen:
        datasets.append(load_queen_dataset(data_dir, feature_cache_dir))
    if zweieck:
        datasets.append(load_zweieck_dataset(data_dir, feature_cache_dir))
    if robbie:
        datasets.append(load_robbie_dataset(data_dir, feature_cache_dir))
    if billboard:
        datasets.append(load_billboard_dataset(data_dir, feature_cache_dir))

    # uses fold 0 for validation, fold 1 for test, rest for training
    train, val, test = dmgr.datasources.get_datasources(
        combine_files(*[ds.get_fold_split() for ds in datasets]),
        **kwargs
    )

    return train, val, test, sum((ds.gt_files for ds in datasets), [])
