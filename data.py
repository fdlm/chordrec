import madmom as mm
import numpy as np
import string


FPS = 50


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
    return mm.audio.spectrogram.LogarithmicFilteredSpectrogram(
        audio_file, fps=FPS
    ).astype(np.float32)

