"""
extract_perfect_chroma.py

    Computes "perfect" chroma vectors based on the ground truth chord
    annotations of a file.

Usage:
    extract_perfect_chroma.py [options] <fps> <dirs>...

Arguments:
    <fps>   frames per second
    <dirs>  directories containing ground truth and audio files.
            audio files are needed for song length

Options:
    -o=<output_dir>  where to put the resulting chromas
                     [default: ./feature_cache]
"""

from os.path import splitext, basename, join
import numpy as np
from itertools import chain, izip
from docopt import docopt
import madmom as mm
import mir_eval

from dmgr.files import find, match_files


def to_chroma(intervals, labels, num_frames, fps):
    roots, bitmaps, _ = mir_eval.chord.encode_many(labels)
    chromas = mir_eval.chord.rotate_bitmaps_to_roots(bitmaps, roots)
    starts = intervals[:, 0]
    ends = intervals[:, 1]

    # add dummy events
    starts = np.hstack(([-np.inf], starts, ends[-1]))
    ends = np.hstack((starts[1], ends, [np.inf]))
    chromas = np.vstack((np.zeros(12), chromas, np.zeros(12)))

    # Finally, we create the chroma vectors per frame!
    frame_times = np.arange(num_frames, dtype=np.float) / fps

    # IMPORTANT: round everything to milliseconds to prevent errors caused
    # by floating point hell. Ideally, we would round everything to
    # possible *frame times*, but it is easier this way.
    starts = np.round(starts, decimals=3)
    ends = np.round(ends, decimals=3)
    frame_times = np.round(frame_times, decimals=3)

    target_per_frame = ((starts <= frame_times[:, np.newaxis]) &
                        (frame_times[:, np.newaxis] < ends))

    # make sure each frame is assigned to only one target vector
    assert (target_per_frame.sum(axis=1) == 1).all()

    # create the one hot vectors per frame
    return chromas[np.nonzero(target_per_frame)[1]].astype(np.float32)


def main():
    args = docopt(__doc__)

    chord_files = list(chain.from_iterable(
        find(d, '*.chords') for d in args['<dirs>']))
    audio_files = list(chain.from_iterable(
        find(d, '*.flac') for d in args['<dirs>']))

    if len(chord_files) != len(audio_files):
        print 'ERROR: {} chord files, but {} audio files'.format(
            len(chord_files), len(audio_files))

    audio_files = match_files(chord_files, audio_files, '.chords', '.flac')

    for cf, af in izip(chord_files, audio_files):
        sig = mm.audio.signal.FramedSignal(af, fps=float(args['<fps>']))
        intervals, labels = mir_eval.io.load_labeled_intervals(cf)

        chromas = to_chroma(intervals, labels, sig.num_frames, 
                            float(args['<fps>']))

        chroma_file = splitext(basename(cf))[0] + '.features.npy'
        np.save(join(args['-o'], chroma_file), chromas)


if __name__ == '__main__':
    main()
