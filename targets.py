import numpy as np
import string


class TimeAnnotationTarget(object):

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

    def __call__(self, target_file, num_frames):
        """
        Creates one-hot encodings from an annotation file.

        :param target_file: file containing time annotations
        :param num_frames:  number of frames in the audio file
        :return:            one-hot ground truth per frame
        """
        ann = np.loadtxt(target_file, dtype=str)
        class_id = self._annotations_to_targets(ann)
        n_chords = len(class_id)
        # we will add a dummy 'NO CHORD' at the end and at the beginning,
        # because some annotations miss it, are not exactly aligned at the end
        # or do not start at the beginning of an audio file
        one_hot = np.zeros((n_chords + 2, self.num_classes), dtype=np.int32)
        one_hot[np.arange(n_chords) + 1, class_id] = 1

        # these are the dummy 'NO CHORD' annotations. the 'NO CHORD' class is
        # always the last id
        one_hot[0, -1] = 1
        one_hot[-1, -1] = 1

        # make sure everything is in its place
        assert (one_hot.argmax(axis=1)[1:-1] == class_id).all()
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
        return one_hot[np.nonzero(target_per_frame)[1]].astype(np.float32)

    def write_chord_predictions(self, filename, predictions):
        with open(filename, 'w') as f:
            f.writelines(['{:.3f}\t{:.3f}\t{}\n'.format(*p)
                          for p in self._targets_to_annotations(predictions)])


class ChordsMajMin(TimeAnnotationTarget):

    def __init__(self, fps):
        # 25 classes - 12 minor, 12 major, one "No Chord"
        super(ChordsMajMin, self).__init__(fps, 25)

    @property
    def name(self):
        return 'chords_majmin_fps={}'.format(self.fps)

    def _annotations_to_targets(self, annotations):
        """
        Maps chord annotations to 25 classes (12 major, 12 minor, 1 no chord)

        :param annotations: chord annotations
        :return: class id per annotation
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
        chord_names = annotations[:, -1]
        chord_root_notes = [c.split(':')[0].split('/')[0] for c in chord_names]
        chord_root_note_ids = np.array([root_note_map[crn]
                                        for crn in chord_root_notes])

        # then, map the chords to major and minor. we assume chords with a
        # minor third as first interval are considered minor chords,
        # the rest are major chords, following MIREX, as stated in
        # Taemin Cho, Juan Bello: "On the relative importance of Individual
        # Components of Chord Recognition Systems"

        chord_type = [c.split(':')[1] if ':' in c else '' for c in chord_names]

        # we will shift the class ids for all minor notes by 12
        # (num major chords)
        chord_type_shift = np.array(
            map(lambda x: 12 if 'min' in x or 'dim' in x else 0, chord_type)
        )

        # now we can compute the final chord class id
        return chord_root_note_ids + chord_type_shift

    def _targets_to_annotations(self, targets):
        natural = zip([0, 2, 3, 5, 7, 8, 10], string.uppercase[:7])
        sharp = map(lambda v: ((v[0] + 1) % 12, v[1] + '#'), natural)

        semitone_to_label = dict(sharp + natural)

        def pred_to_cl(pred):
            if pred == 24:
                return 'N'
            return '{}:{}'.format(semitone_to_label[pred % 12],
                                  'maj' if pred < 12 else 'min')

        spf = 1. / self.fps
        labels = [(i * spf, pred_to_cl(p)) for i, p in enumerate(targets)]

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


class ChordsRoot(TimeAnnotationTarget):

    def __init__(self, fps):
        # 13 classes - 12 semitones and "no chord"
        super(ChordsRoot, self).__init__(fps, 13)

    @property
    def name(self):
        return 'chords_root_fps={}'.format(self.fps)

    def _annotations_to_targets(self, annotations):
        """
        Maps chord annotations to 13 classes (12 root tones, 1 no chord)

        :param annotations: chord annotations
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
        chord_names = annotations[:, -1]
        chord_root_notes = [c.split(':')[0].split('/')[0] for c in chord_names]
        chord_root_note_ids = np.array([root_note_map[crn]
                                        for crn in chord_root_notes])

        return chord_root_note_ids

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


def create_target(fps, config):
    return globals()[config['name']](fps=fps, **config['params'])

