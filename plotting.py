import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import madmom as mm
import data


def plot_crf_params(pi, tau, c, A, W, bin_frequencies, fig=None):

    chords = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#',
              'a', 'a#', 'b', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#',
              'N']

    cmap = plt.get_cmap('RdBu_r')
    fig = fig or plt.figure(figsize=(10, 15))
    gs = gridspec.GridSpec(3, 3,
                           width_ratios=[2, 25, 2], height_ratios=[2, 25, 25])

    # plot A, transitions
    ax = fig.add_subplot(gs[4])
    ax.imshow(A, interpolation='none', cmap=cmap)
    ax.set_xticks(())
    ax.set_yticks(())

    # plot c, label bias
    ax = fig.add_subplot(gs[1])
    ax.imshow(c[None, :], interpolation='none', cmap=cmap)
    ax.set_xticks(range(25))
    ax.set_xticklabels(chords)
    ax.set_yticks(())
    xa = ax.get_xaxis()
    xa.set_ticks_position('top')

    # plot pi
    ax = fig.add_subplot(gs[3])
    ax.imshow(pi[:, None], interpolation='none', cmap=cmap)
    ax.set_xticks(())
    ax.set_yticks(range(25))
    ax.set_yticklabels(chords)

    # plot tau
    ax = fig.add_subplot(gs[5])
    ax.imshow(tau[:, None], interpolation='none', cmap=cmap)
    ax.set_xticks(())
    ax.set_yticks(range(25))
    ax.set_yticklabels(chords)
    ya = ax.get_yaxis()
    ya.set_ticks_position('right')

    # plot W
    context = W.shape[0] / len(bin_frequencies)
    ax = fig.add_subplot(gs[7])
    ax.imshow(W.T.reshape(25*context, -1).T, aspect='auto',
              interpolation='none',
              cmap=cmap, origin='lower')
    ax.set_xticks(np.arange(0, 25*context, context) + (context / 2))
    ax.set_xticklabels(chords)

    # add spectrogram ticks to W
    notes = np.array([24, 26, 28, 29, 31, 33, 35])  # MIDI note for C, D, .. B
    notes = np.concatenate([notes + 12 * i for i in range(10)])  # span octaves
    note_freq = mm.audio.filters.midi2hz(notes)
    note_freq = note_freq[note_freq < bin_frequencies.max()]  # cut too high fs
    spec_ticks = np.interp(note_freq, bin_frequencies,
                           range(len(bin_frequencies)))

    ax.set_yticks(spec_ticks)
    ax.tick_params(axis='y', direction='out', length=6, left='on', right='on',
                   labelright='on', width=1)
    ax.set_yticklabels(['c', 'd', 'e', 'f', 'g', 'a', 'b  '] * 7, fontsize=8)

    fig.subplots_adjust(left=0.125, right=0.875, top=0.95, bottom=0.05,
                        wspace=0.05, hspace=0.05)

    return fig


class CrfPlotter:

    def __init__(self, crf, filename):
        import seaborn as sns
        from matplotlib.backends.backend_pdf import PdfPages
        sns.set(style='white')
        self.crf = crf
        self.pdf = PdfPages(filename)
        self.bin_freq = mm.audio.filters.LogFilterbank(
                mm.audio.stft.fft_frequencies(data.LFS_FFTS[0] / 2, 44100),
                num_bands=data.LFS_NUM_BANDS, fmax=data.LFS_FMAX,
                unique_filters=data.LFS_UNIQUE_FILTERS).center_frequencies

    def __call__(self, _):
        fig = plot_crf_params(
            self.crf.pi.get_value(),
            self.crf.tau.get_value(),
            self.crf.c.get_value(),
            self.crf.A.get_value(),
            self.crf.W.get_value(),
            self.bin_freq
        )
        self.pdf.savefig(fig)
        del fig

    def close(self):
        self.pdf.close()

