import os
import madmom as mm
import numpy as np
import string
import dmgr


FPS = 50
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
    # and minor chords
    root_note_map = dict(natural + sharp + flat + [('N', 24)])

    # then, we load the annotations, map the chords to class ids, and finally
    # map class ids to a one-hot encoding. first, map the root notes.
    ann = np.loadtxt(target_file, dtype=str)
    chord_names = ann[:, -1]
    chord_root_notes = [c.split(':')[0].split('/')[0] for c in chord_names]
    chord_root_note_ids = np.array([root_note_map[crn]
                                    for crn in chord_root_notes])

    # then, map the chords to major and minor. we assume everything major
    # except when annotated as 'min' or as 'sus2' (because the 'third' in sus2
    # is closer to a minor third... TODO: check if this makes sense at all!

    chord_type = [c.split(':')[1] if ':' in c else '' for c in chord_names]

    # we will shift the class ids for all minor notes by 12 (num major chords)
    chord_type_shift = np.array(
        map(lambda x: 12 if 'min' in x or 'sus2' in x else 0, chord_type)
    )

    # now we can compute the final chord class id
    chord_class_id = chord_root_note_ids + chord_type_shift

    n_chords = len(chord_class_id)
    # 25 classes - 12 major, 12 minor, one no chord
    one_hot = np.zeros((n_chords, 25), dtype=np.int32)
    one_hot[np.arange(n_chords), chord_class_id] = 1

    # make sure everything is in its place
    assert (one_hot.argmax(axis=1) == chord_class_id).all()
    assert (one_hot.sum(axis=1) == 1).all()

    # Now, we create the time stamps. if no explicit end times are given,
    # we take the start time of the next chord as end time for the current.
    start = ann[:, 0].astype(np.float)
    end = (ann[:, 1].astype(np.float) if ann.shape[1] > 2 else
           np.hstack((start[1:], [np.inf])))

    # Finally, we create the one-hot encoding per frame!
    frame_times = np.arange(num_frames) * (1. / fps)
    target_per_frame = ((start <= frame_times[:, np.newaxis]) &
                        (frame_times[:, np.newaxis] < end))

    # make sure each frame is assigned to only one target vector
    assert (target_per_frame.sum(axis=1) == 1).all()

    # create the one hot vectors per frame
    return one_hot[np.nonzero(target_per_frame)[1]]


def compute_features(audio_file):
    """
    This function just computes the features for an audio file
    :param audio_file: audio file to compute the features for
    :return: features as numpy array (or similar)
    """
    specs = [
        mm.audio.spectrogram.LogarithmicFilteredSpectrogram(
            audio_file, fps=FPS, num_bands=12, frame_size=ffts)
        for ffts in [1024, 2048, 4096]
    ]

    return np.hstack(specs).astype(np.float32)


def get_whitened_context_datasources(files):
    train_set = dmgr.datasources.AggregatedDataSource.from_files(
        files['train']['feat'], files['train']['targ'], memory_mapped=True,
        data_source_type=dmgr.datasources.ContextDataSource,
        context_size=3
    )

    val_set = dmgr.datasources.AggregatedDataSource.from_files(
        files['val']['feat'], files['val']['targ'], memory_mapped=True,
        data_source_type=dmgr.datasources.ContextDataSource,
        context_size=3
    )

    test_set = dmgr.datasources.AggregatedDataSource.from_files(
        files['test']['feat'], files['test']['targ'], memory_mapped=True,
        data_source_type=dmgr.datasources.ContextDataSource,
        context_size=3
    )

    preproc = dmgr.preprocessing.DataWhitener()
    preproc.train(train_set, batch_size=4096)

    train_set = dmgr.datasources.PreProcessedDataSource(
        train_set, preproc
    )

    val_set = dmgr.datasources.PreProcessedDataSource(
        val_set, preproc
    )

    test_set = dmgr.datasources.PreProcessedDataSource(
        test_set, preproc
    )

    return train_set, val_set, test_set


class Beatles:

    def __init__(self, data_dir=DATA_DIR, feature_cache_dir=CACHE_DIR):
        data_dir = os.path.join(data_dir, 'beatles')
        feature_cache_dir = os.path.join(feature_cache_dir, 'beatles')

        src_files = dmgr.files.expand(data_dir, '*' + SRC_EXT)
        gt_files = dmgr.files.expand(data_dir, '*' + GT_EXT)
        gt_files = dmgr.files.match_files(src_files, gt_files,
                                          SRC_EXT, GT_EXT)

        feat_files, target_files = dmgr.files.prepare(
            src_files, gt_files, feature_cache_dir,
            compute_feat=compute_features,
            compute_targets=compute_targets,
            fps=FPS
        )

        self.feature_files = feat_files
        self.target_files = target_files

        split_dir = os.path.join(data_dir, 'splits')

        self.split_defs = [
            os.path.join(split_dir,
                         '8-fold_cv_album_distributed_{}.fold'.format(f))
            for f in range(8)
        ]

    def get_fold_split(self, val_fold=0, test_fold=1):
        train_feat, val_feat, test_feat = \
            dmgr.files.predefined_train_val_test_split(
                self.feature_files,
                self.split_defs[val_fold],
                self.split_defs[test_fold],
                match_suffix=dmgr.files.FEAT_EXT
            )

        train_targ = dmgr.files.match_files(train_feat, self.target_files,
                                            dmgr.files.FEAT_EXT,
                                            dmgr.files.TARGET_EXT)
        val_targ = dmgr.files.match_files(val_feat, self.target_files,
                                          dmgr.files.FEAT_EXT,
                                          dmgr.files.TARGET_EXT)
        test_targ = dmgr.files.match_files(test_feat, self.target_files,
                                           dmgr.files.FEAT_EXT,
                                           dmgr.files.TARGET_EXT)

        return {'train': {'feat': train_feat,
                          'targ': train_targ},
                'val': {'feat': val_feat,
                        'targ': val_targ},
                'test': {'feat': test_feat,
                         'targ': test_targ}}
