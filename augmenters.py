import numpy as np
from scipy.ndimage import shift
import random
from targets import one_hot


class SemitoneShift(object):

    def __init__(self, p, max_shift, bins_per_semitone):
        """
        Augmenter that shifts by semitones a spectrum with logarithmically
        spaced frequency bins.

        :param p: percentage of data to be shifted
        :param max_shift: maximum number of semitones to shift
        :param bins_per_semitone: number of spectrogram bins per semitone
        """
        self.p = p
        self.max_shift = max_shift
        self.bins_per_semitone = bins_per_semitone

    def __call__(self, batch_iterator):
        """
        :param batch_iterator: data iterator that yields the data to be
                               augmented
        :return: augmented data/target pairs
        """

        for data, targets in batch_iterator:
            batch_size = len(data)

            shifts = np.random.randint(-self.max_shift,
                                       self.max_shift + 1, batch_size)

            # zero out shifts for 1-p percentage
            no_shift = random.sample(range(batch_size),
                                     int(batch_size * (1 - self.p)))
            shifts[no_shift] = 0

            chord_classes = targets.argmax(-1)
            no_chord_class = targets.shape[-1] - 1
            no_chords = (chord_classes == no_chord_class)
            chord_roots = chord_classes % 12
            chord_majmin = chord_classes / 12

            new_chord_roots = (chord_roots + shifts) % 12
            new_chord_classes = new_chord_roots + chord_majmin * 12
            new_chord_classes[no_chords] = no_chord_class
            new_targets = one_hot(new_chord_classes, no_chord_class + 1)

            new_data = np.empty_like(data)
            for i in range(batch_size):
                # TODO: remove data from upper and lower parts that got
                #       rolled (?)
                new_data[i] = np.roll(
                    data[i], shifts[i] * self.bins_per_semitone, axis=-1)

            yield new_data, new_targets


class Detuning(object):

    def __init__(self, p, max_shift, bins_per_semitone):
        """
        Augmenter that shifts a spectrogram with logarithmically spaced
        frequency bins by maximum 0.5 semitones
        :param p: percentage of data to be shifted
        :param max_shift: maximum fraction of semitone to shirt (<= 0.5)
        :param bins_per_semitone: number of spectrogram bins per semitone
        """
        if max_shift >= 0.5:
            raise ValueError('Detuning only works up to half a semitone!')
        self.p = p
        self.max_shift = max_shift
        self.bins_per_semitone = bins_per_semitone

    def __call__(self, batch_iterator):
        """
        :param batch_iterator: data iterator that yields the data to be
                               augmented
        :return: augmented data/target pairs
        """
        for data, targets in batch_iterator:
            batch_size = len(data)

            shifts = np.random.rand(batch_size) * 2 * self.max_shift - \
                self.max_shift

            # zero out shifts for 1-p percentage
            no_shift = random.sample(range(batch_size),
                                     int(batch_size * (1 - self.p)))
            shifts[no_shift] = 0

            new_data = np.empty_like(data)
            for i in range(batch_size):
                new_data[i] = shift(
                    data[i], (shifts[i] * self.bins_per_semitone, 0))

            yield new_data, targets


def create_augmenters(augmentation):
    return [globals()[name](**params)
            for name, params in augmentation.iteritems()]
