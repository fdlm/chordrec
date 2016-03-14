import mir_eval
import numpy as np
import madmom as mm
import dmgr
from scipy.spatial.distance import cdist
from exp_utils import TempDir
import targets
import os
import test
import sys

USAGE = """
Usage:
    test_binary_pattern.py chroma <chroma_dir> audio <audio_dir>
                           ann <annotation_dir> dst <dst_dir>
"""


def create_binary_patterns():
    _, maj_bitmap, _ = mir_eval.chord.encode('C:maj')
    a_rt, min_bitmap, _ = mir_eval.chord.encode('A:min')
    roots = np.roll(np.arange(12), -a_rt)
    maj_bitmaps = np.tile(maj_bitmap, (12, 1))
    min_bitmaps = np.tile(min_bitmap, (12, 1))
    maj_chroma = mir_eval.chord.rotate_bitmaps_to_roots(maj_bitmaps, roots)
    min_chroma = mir_eval.chord.rotate_bitmaps_to_roots(min_bitmaps, roots)

    patterns = np.vstack((maj_chroma, min_chroma))
    return patterns


def main():
    if len(sys.argv) < 8:
        print USAGE
        return 1

    chroma_dirs = []
    audio_dirs = []
    ann_dirs = []
    dst_dir = []

    cur_lst = None
    for arg in sys.argv[1:]:
        if arg == 'chroma':
            cur_lst = chroma_dirs
        elif arg == 'audio':
            cur_lst = audio_dirs
        elif arg == 'ann':
            cur_lst = ann_dirs
        elif arg == 'dst':
            cur_lst = dst_dir
        else:
            cur_lst.append(arg)

    chroma_files = dmgr.files.expand(chroma_dirs, '*.features.npy')
    audio_files = dmgr.files.expand(audio_dirs, '*.flac')
    annotation_files = dmgr.files.expand(ann_dirs, '*.chords')

    audio_files = dmgr.files.match_files(chroma_files, audio_files,
                                         '.features.npy', '.flac')
    annotation_files = dmgr.files.match_files(chroma_files, annotation_files,
                                              '.features.npy', '.chords')

    patterns = create_binary_patterns()
    fps = 9.98641304347826086957
    target = targets.ChordsMajMin(fps=fps)

    pred_files = []

    with TempDir() as tmpdir:
        for cf, af in zip(chroma_files, audio_files):
            chroma = np.load(cf)
            pred = cdist(patterns, chroma, metric='euclidean').argmin(axis=0)

            # find no-chords based on spl smaller than -57 dB
            audio = mm.audio.signal.FramedSignal(af, frame_size=4096, fps=fps,
                                                 num_channels=1)
            spl = np.array([mm.audio.signal.sound_pressure_level(frame)
                            for frame in audio])
            pred[:len(spl)][spl < -57] = 24
            pred[len(spl):] = 24

            fn = os.path.splitext(os.path.basename(af))[0]
            pred_file = os.path.join(tmpdir, fn + '.chords.txt')
            target.write_chord_predictions(pred_file, pred)

            pred_files.append(pred_file)

        scores = test.compute_average_scores(annotation_files, pred_files)
        test.print_scores(scores)


if __name__ == '__main__':
    main()
