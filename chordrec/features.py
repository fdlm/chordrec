import numpy as np
import madmom as mm
import pickle


class ConstantQ:

    def __init__(self, num_bands, fmin, num_octaves, fps, align, log_div,
                 sample_rate=44100, fold=None):

        self.fps = fps
        self.num_bands = num_bands
        self.align = align
        self.fmin = fmin
        self.num_octaves = num_octaves
        self.log_div = log_div

        self.sample_rate = sample_rate

        from yaafelib import FeaturePlan, Engine

        fp = FeaturePlan(sample_rate=sample_rate)

        cqt_config = " ".join(['cqt: CQT',
                               'CQTAlign={}'.format(align),
                               'CQTBinsPerOctave={}'.format(num_bands),
                               'CQTMinFreq={}'.format(fmin),
                               'CQTNbOctaves={}'.format(num_octaves),
                               'stepSize={}'.format(sample_rate / fps)
                               ])

        fp.addFeature(cqt_config)

        df = fp.getDataFlow()
        self.engine = Engine()
        self.engine.load(df)

    @property
    def name(self):
        return 'cqt_fps={}_num-bands={}_align={}_fmin={}_num_oct={}'\
               '_logdiv={}'.format(self.fps, self.num_bands, self.align,
                                   self.fmin, self.num_octaves, self.log_div)

    def __call__(self, audio_file):

        audio = mm.audio.signal.Signal(audio_file,
                                       sample_rate=self.sample_rate,
                                       num_channels=1).astype(np.float64)

        cqt = self.engine.processAudio(audio.reshape((1, -1)))['cqt']
        # compensate for different padding in madmom vs. yaafe and convert
        # to float32
        cqt = np.vstack((cqt, np.zeros(cqt.shape[1:]))).astype(np.float32)

        if self.log_div:
            return np.log(cqt / self.log_div + 1)
        else:
            return cqt


class LogFiltSpec:

    def __init__(self, frame_sizes, num_bands, fmin, fmax, fps, unique_filters,
                 sample_rate=44100, fold=None):

        self.frame_sizes = frame_sizes
        self.num_bands = num_bands
        self.fmax = fmax
        self.fmin = fmin
        self.fps = fps
        self.unique_filters = unique_filters
        self.sample_rate = sample_rate

    @property
    def name(self):
        return 'lfs_fps={}_num-bands={}_fmin={}_fmax={}_frame_sizes=[{}]'.format(
                self.fps, self.num_bands, self.fmin, self.fmax,
                '-'.join(map(str, self.frame_sizes))
        ) + ('_uf' if self.unique_filters else '')

    def __call__(self, audio_file):
        # do not resample because ffmpeg/avconv creates terrible sampling
        # artifacts
        specs = [
            mm.audio.spectrogram.LogarithmicFilteredSpectrogram(
                audio_file, num_channels=1, sample_rate=self.sample_rate,
                fps=self.fps, frame_size=ffts,
                num_bands=self.num_bands, fmin=self.fmin, fmax=self.fmax,
                unique_filters=self.unique_filters)
            for ffts in self.frame_sizes
        ]

        return np.hstack(specs).astype(np.float32)


class Chroma:

    def __init__(self, frame_size, fmax, fps, oct_width, center_note, log_eta,
                 sample_rate=44100, fold=None):
        self.fps = fps
        self.fmax = fmax
        self.sample_rate = sample_rate
        self.oct_width = oct_width
        self.center_note = center_note
        self.frame_size = frame_size
        self.log_eta = log_eta

        # parameters are based on Cho and Bello, 2014.
        import librosa
        ctroct = (librosa.hz_to_octs(librosa.note_to_hz(center_note))
                  if center_note is not None else None)

        self.filterbank = librosa.filters.chroma(
            sr=sample_rate, n_fft=frame_size, octwidth=oct_width,
            ctroct=ctroct).T[:-1]

        # mask out everything above fmax
        from bottleneck import move_mean
        m = np.fft.fftfreq(
            frame_size, 1. / sample_rate)[:frame_size / 2] < fmax
        mask_smooth = move_mean(m, window=10, min_count=1)
        self.filterbank *= mask_smooth[:, np.newaxis]

    @property
    def name(self):
        if self.oct_width is not None:
            gauss_str = '_octwidth={:g}_cnote={}'.format(self.oct_width,
                                                         self.center_note)
        else:
            gauss_str = ''

        if self.log_eta is not None:
            log_str = '_log={}'.format(self.log_eta)
        else:
            log_str = ''

        return 'chroma_fps={}_fmax={}_frame_size={}'.format(
            self.fps, self.fmax, self.frame_size) + gauss_str + log_str

    def __call__(self, audio_file):
        spec = mm.audio.spectrogram.Spectrogram(
            audio_file, num_channels=1, sample_rate=self.sample_rate,
            fps=self.fps, frame_size=4096,
        )

        if self.log_eta is not None:
            spec = np.log(self.log_eta * spec / spec.max() + 1)

        chroma = np.dot(spec, self.filterbank)
        norm = np.sqrt(np.sum(chroma ** 2, axis=1))
        norm[norm < 1e-20] = 1.
        return (chroma / norm[:, np.newaxis]).astype(np.float32)


class ChromaCq:

    def __init__(self, fps, win_center, win_width, log_eta,
                 sample_rate=44100, fold=None):
        """
        Computes Chromas from a constant q transform.
        :param fps:          frames per second
        :param win_center:   midi number of window center note
        :param win_width:    width of weighting window
        :param log_eta:      scaling parameter for log
        :param sample_rate:  sample rate of the audio
        """
        self.fps = fps
        self.sample_rate = sample_rate
        self.num_bins = 84
        self.log_eta = log_eta

        if win_center is None:
            self.win = None
            self.win_center = None
            self.win_width = None
        else:
            # cq spec starts at C1, which is midi pitch 24. the zeroth bin thus
            # corresponds to midi note 24, and we have to adjust win_center
            self.win_center = float(win_center - 24)
            self.win_width = float(win_width)
            self.win = np.exp(
                -0.5 * ((self.win_center - np.arange(self.num_bins)) /
                        self.win_width) ** 2
            )

    @property
    def name(self):
        if self.win is not None:
            win_str = '_winc={}_winw={}'.format(self.win_center,
                                                self.win_width)
        else:
            win_str = ''

        log_str = '_log_eta={}'.format(self.log_eta) if self.log_eta else ''
        return 'chroma_cq_fps={}'.format(self.fps) + win_str + log_str

    def __call__(self, audio_file):
        import librosa
        y = mm.audio.signal.Signal(audio_file, num_channels=1,
                                   sample_rate=self.sample_rate)

        cq = librosa.core.cqt(y, sr=y.sample_rate, tuning=0,
                              fmin=mm.audio.filters.midi2hz(24),
                              n_bins=self.num_bins,
                              hop_length=int(self.sample_rate / self.fps))

        if self.log_eta is not None:
            cq = np.log(self.log_eta * cq / cq.max() + 1)

        if self.win is not None:
            cq *= self.win[:, np.newaxis]

        return librosa.feature.chroma_cqt(y=None, C=cq, tuning=0,
                                          norm=2).T.astype(np.float32)


class HarmonicPitchClassProfile:

    def __init__(self, fps, frame_size, fmax, num_bands,
                 sample_rate=44100, fold=None):
        self.fps = fps
        self.frame_size = frame_size
        self.fmax = fmax
        self.sample_rate = sample_rate
        self.num_bands = num_bands

    @property
    def name(self):
        return 'hpcp_fps={}_fmax={}_nbands={}_frame_size={}'.format(
            self.fps, self.fmax, self.num_bands, self.frame_size
        )

    def __call__(self, audio_file):
        from madmom.audio import chroma

        hpcp = chroma.HarmonicPitchClassProfile(
            audio_file, fps=self.fps, fmax=self.fmax,
            num_classes=self.num_bands, sample_rate=self.sample_rate
        )

        norm = np.sqrt(np.sum(hpcp ** 2, axis=1))
        norm[norm < 1e-20] = 1.
        return (hpcp / norm[:, np.newaxis]).astype(np.float32)


class DeepChroma:

    def __init__(self, fps, fmin=65, fmax=2100, unique_filters=True,
                 models=None, sample_rate=44100, fold=None):
        assert fps == 10, 'Cannot handle fps different from 10 yet.'
        from madmom.audio.chroma import DeepChromaProcessor
        from hashlib import sha1
        self.fps = fps
        self.fmin = fmin
        self.fmax = fmax
        self.unique_filters = unique_filters
        self.dcp = DeepChromaProcessor(
            fmin=fmin, fmax=fmax, unique_filters=unique_filters, models=models
        )
        self.model_hash = sha1(pickle.dumps(self.dcp)).hexdigest()

    @property
    def name(self):
        return 'deepchroma_fps={}_fmin={}_fmax={}_uf={}_mdlhsh={}'.format(
            self.fps, self.fmin, self.fmax, self.unique_filters,
            self.model_hash
        )

    def __call__(self, audio_file):
        return self.dcp(audio_file)


class PrecomputedFeature:

    def __init__(self, name, fps, fold):
        self._name = name
        self.fps = fps
        self.fold = fold

    @property
    def name(self):
        return self._name.format(fps=self.fps, fold=self.fold)

    def __call__(self, audio_file):
        raise NotImplementedError(
            'Cannot compute features for {}. '
            'This feature is only precomputed!'.format(audio_file))


def add_sacred_config(ex):
    ex.add_named_config(
        'constant_q',
        feature_extractor=dict(
            name='ConstantQ',
            params=dict(
                fps=10,
                num_bands=24,
                fmin=30,
                num_octaves=8,
                log_div=500.,
                align='c'
            )
        )
    )

    ex.add_named_config(
        'log_filt_spec',
        feature_extractor=dict(
            name='LogFiltSpec',
            params=dict(
                fps=10,
                frame_sizes=[8192],
                num_bands=24,
                fmin=65,
                fmax=2100,
                unique_filters=True,
            )
        )
    )

    ex.add_named_config(
        'chroma_clp',
        feature_extractor=dict(
            name='PrecomputedFeature',
            params=dict(
                name='chroma_clp_fps={fps}',
                fps=10,
            )
        )
    )

    ex.add_named_config(
        'perfect_chroma',
        feature_extractor=dict(
            name='PrecomputedFeature',
            params=dict(
                name='perfect_chroma_fps={fps}',
                fps=10
            )
        )
    )

    ex.add_named_config(
        'gap_feature',
        feature_extractor=dict(
            name='PrecomputedFeature',
            params=dict(
                name='gap_feature/features_fold_{fold}',
                fps=10,
            )
        )
    )

    ex.add_named_config(
        'deep_chroma_pc',
        feature_extractor=dict(
            name='PrecomputedFeature',
            params=dict(
                name='deep_chroma_pc',
                fps=10
            )
        )
    )

    ex.add_named_config(
        'deep_chroma',
        feature_extractor=dict(
            name='DeepChroma',
            params=dict(
                fps=10
            )
        )
    )

    ex.add_named_config(
        'hpcp',
        feature_extractor=dict(
            name='HarmonicPitchClassProfile',
            params=dict(
                fps=10,
                frame_size=8192,
                fmax=5500,
                num_bands=36,
            )
        )
    )

    ex.add_named_config(
        'chroma_hpcp',
        feature_extractor=dict(
            name='HarmonicPitchClassProfile',
            params=dict(
                fps=10,
                frame_size=8192,
                fmax=5500,
                num_bands=12,
            )
        )
    )

    ex.add_named_config(
        'chroma',
        feature_extractor=dict(
            name='Chroma',
            params=dict(
                fps=10,
                frame_size=4096,
                fmax=5500,
                oct_width=None,
                center_note=None,
                log_eta=None
            )
        )
    )

    ex.add_named_config(
        'chroma_w_log',
        feature_extractor=dict(
            name='Chroma',
            params=dict(
                fps=10,
                frame_size=4096,
                fmax=5500,
                oct_width=15./12,
                center_note='C4',
                log_eta=1000
            )
        )
    )

    ex.add_named_config(
        'chroma_cq',
        feature_extractor=dict(
            name='ChromaCq',
            params=dict(
                fps=9.98641304347826086957,
                win_center=None,
                win_width=None,
                log_eta=None
            )
        )
    )

    ex.add_named_config(
        'chroma_cq_w_log',
        feature_extractor=dict(
            name='ChromaCq',
            params=dict(
                fps=9.98641304347826086957,
                # paramters taken from Cho's paper
                win_center=60,
                win_width=15,
                log_eta=1000
            )
        )
    )


def create_extractor(config, fold):
    return globals()[config['name']](fold=fold, **config['params'])
