import numpy as np
import string
import mir_eval


def one_hot(class_ids, num_classes):
    """
    Create one-hot encoding of class ids
    :param class_ids:   ids of classes to map
    :param num_classes: number of classes
    :return: one-hot encoding of class ids
    """
    oh = np.zeros((len(class_ids), num_classes), dtype=np.float32)
    oh[np.arange(len(class_ids)), class_ids] = 1

    # make sure one-hot encoding corresponds to class ids
    assert (oh.argmax(axis=1) == class_ids).all()
    # make sure there is only one id set per vector
    assert (oh.sum(axis=1) == 1).all()

    return oh


class IntervalAnnotationTarget(object):

    def __init__(self, fps, num_classes):
        self.fps = fps
        self.num_classes = num_classes

    def _annotations_to_targets(self, annotations):
        """
        Class ID of 'no chord' should always be last!
        :param annotations:
        :return:
        """
        raise NotImplementedError('Implement this')

    def _targets_to_annotations(self, targets):
        raise NotImplementedError('Implement this.')

    def _dummy_target(self):
        raise NotImplementedError('Implement this.')

    def __call__(self, target_file, num_frames=None):
        """
        Creates one-hot encodings from an annotation file.

        :param target_file: file containing time annotations
        :param num_frames:  number of frames in the audio file. if None,
                            estimate from the end of last annotation
        :return:            one-hot ground truth per frame
        """
        ann = np.loadtxt(target_file,
                         comments=None,
                         dtype=[('start', np.float),
                                ('end', np.float),
                                # assumes chord descriptions are
                                # shorter than 50 characters
                                ('label', 'S50')])

        if num_frames is None:
            num_frames = np.ceil(ann['end'][-1] * self.fps)

        # we will add a dummy class at the end and at the beginning,
        # because some annotations miss it, are not exactly aligned at the end
        # or do not start at the beginning of an audio file
        targets = np.vstack((self._dummy_target(),
                             self._annotations_to_targets(ann['label']),
                             self._dummy_target()))

        # add the times for the dummy events
        start = np.hstack(([-np.inf], ann['start'], ann['end'][-1]))
        end = np.hstack((ann['start'][0], ann['end'], [np.inf]))

        # next, we have to assign each frame a target. first, compute the
        # frame times
        frame_times = np.arange(num_frames, dtype=np.float) / self.fps

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
        return targets[np.nonzero(target_per_frame)[1]].astype(np.float32)

    def write_chord_predictions(self, filename, predictions):
        with open(filename, 'w') as f:
            f.writelines(['{:.3f}\t{:.3f}\t{}\n'.format(*p)
                          for p in self._targets_to_annotations(predictions)])


class ChordsMajMin(IntervalAnnotationTarget):

    def __init__(self, fps):
        # 25 classes - 12 minor, 12 major, one "No Chord"
        super(ChordsMajMin, self).__init__(fps, 25)

    @property
    def name(self):
        return 'chords_majmin_fps={}'.format(self.fps)

    def _dummy_target(self):
        dt = np.zeros(self.num_classes, dtype=np.float32)
        dt[-1] = 1
        return dt

    def _annotations_to_targets(self, labels):
        """
        Maps chord annotations to 25 classes (12 major, 12 minor, 1 no chord)

        :param labels: chord labels
        :return: one-hot encoding of class id per annotation
        """
        # first, create chord/class mapping. root note 'A' has id 0, increasing
        # with each semitone. we have duplicate mappings for flat and sharp
        # notes, just to be sure.
        natural = zip(string.uppercase[:7], [0, 2, 3, 5, 7, 8, 10])
        sharp = map(lambda v: (v[0] + '#', (v[1] + 1) % 12), natural)
        flat = map(lambda v: (v[0] + 'b', (v[1] - 1) % 12), natural)

        # 'no chord' is coded as 'N'. The class ID of 'N' is 24, after all
        # major and minor chords. Sometimes there is also an 'X' annotation,
        # meaning that the chord cannot be properly determined on beat-lebel
        # (too much going on in the audio). We will treat this also as
        # 'no chord'
        root_note_map = dict(natural + sharp + flat + [('N', 24), ('X', 24)])

        # then, we load the annotations, map the chords to class ids, and
        # finally map class ids to a one-hot encoding. first, map the root
        # notes.
        chord_root_notes = [c.split(':')[0].split('/')[0] for c in labels]
        chord_root_note_ids = np.array([root_note_map[crn]
                                        for crn in chord_root_notes])

        # then, map the chords to major and minor. we assume chords with a
        # minor third as first interval are considered minor chords,
        # the rest are major chords, following MIREX, as stated in
        # Taemin Cho, Juan Bello: "On the relative importance of Individual
        # Components of Chord Recognition Systems"

        chord_type = [c.split(':')[1] if ':' in c else '' for c in labels]

        # we will shift the class ids for all minor notes by 12
        # (num major chords)
        chord_type_shift = np.array(
            map(lambda x: 12 if 'min' in x or 'dim' in x else 0, chord_type)
        )

        # now we can compute the final chord class id
        return one_hot(chord_root_note_ids + chord_type_shift,
                       self.num_classes)

    def _targets_to_annotations(self, targets):
        natural = zip([0, 2, 3, 5, 7, 8, 10], string.uppercase[:7])
        sharp = map(lambda v: ((v[0] + 1) % 12, v[1] + '#'), natural)

        semitone_to_label = dict(sharp + natural)

        def pred_to_label(pred):
            if pred == 24:
                return 'N'
            return '{}:{}'.format(semitone_to_label[pred % 12],
                                  'maj' if pred < 12 else 'min')

        spf = 1. / self.fps
        labels = [(i * spf, pred_to_label(p)) for i, p in enumerate(targets)]

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


class ChordsRoot(IntervalAnnotationTarget):

    def __init__(self, fps):
        # 13 classes - 12 semitones and "no chord"
        super(ChordsRoot, self).__init__(fps, 13)

    @property
    def name(self):
        return 'chords_root_fps={}'.format(self.fps)

    def _dummy_target(self):
        dt = np.zeros(self.num_classes, dtype=np.float32)
        dt[-1] = 1
        return dt

    def _annotations_to_targets(self, labels):
        """
        Maps chord annotations to 13 classes (12 root tones, 1 no chord)

        :param labels: chord label
        :return: class id per annotation
        """
        # first, create chord/class mapping. root note 'A' has id 0, increasing
        # with each semitone. we have duplicate mappings for flat and sharp
        # notes, just to be sure.
        natural = zip(string.uppercase[:7], [0, 2, 3, 5, 7, 8, 10])
        sharp = map(lambda v: (v[0] + '#', (v[1] + 1) % 12), natural)
        flat = map(lambda v: (v[0] + 'b', (v[1] - 1) % 12), natural)

        # 'no chord' is coded as 'N'. The class ID of 'N' is 12, after all
        # root notes. Sometimes there is also an 'X' annotation,
        # meaning that the chord cannot be properly determined on beat-lebel
        # (too much going on in the audio). We will treat this also as
        # 'no chord'
        root_note_map = dict(natural + sharp + flat + [('N', 12), ('X', 12)])

        # then, we load the annotations, map the chords to class ids, and
        # finally map class ids to a one-hot encoding. first, map the root
        # notes.
        chord_root_notes = [c.split(':')[0].split('/')[0] for c in labels]
        chord_root_note_ids = np.array([root_note_map[crn]
                                        for crn in chord_root_notes])

        return one_hot(chord_root_note_ids, self.num_classes)

    def _targets_to_annotations(self, targets):
        natural = zip([0, 2, 3, 5, 7, 8, 10], string.uppercase[:7])
        sharp = map(lambda v: ((v[0] + 1) % 12, v[1] + '#'), natural)

        semitone_to_label = dict(sharp + natural + [(12, 'N')])
        spf = 1. / self.fps
        labels = [(i * spf, semitone_to_label[p])
                  for i, p in enumerate(targets)]

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


class ChordsMajMinSevenths(IntervalAnnotationTarget):

    def __init__(self, fps):
        # 73 classes - maj, 7, maj7, min, min7 minmaj7 with 12 each, 1 no chord
        super(ChordsMajMinSevenths, self).__init__(fps, 73)

    @property
    def name(self):
        return 'chords_majminsevenths_fps={}'.format(self.fps)

    def _dummy_target(self):
        dt = np.zeros(self.num_classes, dtype=np.float32)
        dt[-1] = 1
        return dt

    def _annotations_to_targets(self, labels):
        root, semis, _ = mir_eval.chord.encode_many(labels, True)
        class_ids = root.copy()

        # 'no chord' is last class
        class_ids[class_ids == -1] = self.num_classes - 1

        # minor chords start at idx 36
        class_ids[semis[:, 3] == 1] += 36

        # seventh shift
        seventh = semis[:, 10] == 1
        maj_seventh = semis[:, 11] == 1

        # this weirdness is necessary because of a B:sus4(b7)/7 annotation
        # in the RWC corpus...
        maj_seventh &= ~seventh
        assert (seventh & maj_seventh).sum() == 0

        class_ids[seventh] += 12
        class_ids[maj_seventh] += 24

        return one_hot(class_ids, self.num_classes)

    def _targets_to_annotations(self, targets):
        natural = zip([0, 2, 3, 5, 7, 8, 10], string.uppercase[:7])
        sharp = map(lambda v: ((v[0] + 1) % 12, v[1] + '#'), natural)
        roots = {(a - 3) % 12: b for a, b in dict(sharp + natural).iteritems()}
        ext = ['maj', '7', 'maj7', 'min', 'min7', 'minmaj7']

        def pred_to_label(pred):
            if pred == self.num_classes - 1:
                return 'N'

            return '{root}:{ext}'.format(
                root=roots[pred % 12],
                ext=ext[pred / 12]
            )

        spf = 1. / self.fps
        labels = [(i * spf, pred_to_label(p)) for i, p in enumerate(targets)]

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


class ChromaTarget(IntervalAnnotationTarget):

    def __init__(self, fps):
        # vector of 12 semitones
        super(ChromaTarget, self).__init__(fps, 12)

    @property
    def name(self):
        return 'chroma_target_fps={}'.format(self.fps)

    def _dummy_target(self):
        return mir_eval.chord.NO_CHORD_ENCODED[1]

    def _annotations_to_targets(self, labels):
        roots, bitmaps, _ = mir_eval.chord.encode_many(labels)
        chromas = mir_eval.chord.rotate_bitmaps_to_roots(bitmaps, roots)
        return chromas

    def _targets_to_annotations(self, targets):
        raise RuntimeError('Does not work with this target.')


def add_sacred_config(ex):
    ex.add_named_config(
        'chords_maj_min',
        target=dict(
            name='ChordsMajMin',
            params=dict()
        )
    )
    ex.add_named_config(
        'chords_root',
        target=dict(
            name='ChordsRoot',
            params=dict()
        )
    )
    ex.add_named_config(
        'chords_maj_min_sevenths',
        target=dict(
            name='ChordsMajMinSevenths',
            params=dict()
        )
    )


def create_target(fps, config):
    return globals()[config['name']](fps=fps, **config['params'])

