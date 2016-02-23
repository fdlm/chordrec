import numpy as np
import madmom as mm


class ConstantQ:

    def __init__(self, num_bands=24, fmin=30, num_octaves=8,
                 fps=10, align='c', sample_rate=44100):

        self.fps = fps
        self.num_bands = num_bands
        self.align = align
        self.fmin = fmin
        self.num_octaves = num_octaves

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
        return 'cqt_fps={}_num-bands={}_align={}_fmin={}_num_oct={}'.format(
            self.fps, self.num_bands, self.align, self.fmin, self.num_octaves
        )

    def __call__(self, audio_file):

        audio = mm.audio.signal.Signal(audio_file,
                                       sample_rate=self.sample_rate,
                                       num_channels=1).astype(np.float64)

        cqt = self.engine.processAudio(audio.reshape((1, -1)))['cqt']
        # compensate for different padding in madmom vs. yaafe and convert
        # to float32
        return np.vstack((cqt, np.zeros(cqt.shape[1:]))).astype(np.float32)


class LogFiltSpec:

    def __init__(self, frame_sizes, num_bands, fmax, fps, unique_filters,
                 sample_rate=44100):

        self.frame_sizes = frame_sizes
        self.num_bands = num_bands
        self.fmax = fmax
        self.fps = fps
        self.unique_filteres = unique_filters
        self.sample_rate = sample_rate

    @property
    def name(self):
        return 'lfs_fps={}_num-bands={}_fmax={}_frame_sizes=[{}]'.format(
                self.fps, self.num_bands, self.fmax,
                '-'.join(map(str, self.frame_sizes))
        )

    def __call__(self, audio_file):
        # do not resample because ffmpeg/avconv creates terrible sampling
        # artifacts
        specs = [
            mm.audio.spectrogram.LogarithmicFilteredSpectrogram(
                audio_file, num_channels=1, sample_rate=self.sample_rate,
                fps=self.fps, frame_size=ffts,
                num_bands=self.num_bands, fmax=self.fmax,
                unique_filters=self.unique_filteres)
            for ffts in self.frame_sizes
        ]

        return np.hstack(specs).astype(np.float32)


class ChromaClp:

    def __init__(self, fps):
        assert fps == 10.022727272727273

    @property
    def name(self):
        return 'chroma_clp100'

    def __call__(self, audio_file):
        # this feature is precompute-only!
        raise NotImplementedError('This feature is only precomputed!')


class PerfectChroma:

    def __init__(self, fps):
        self.fps = fps

    @property
    def name(self):
        return 'perfect_chroma_fps={}'.format(self.fps)

    def __call__(self, audio_file):
        # this feature is precompute-only
        raise NotImplementedError('This feature is only precomputed')


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
                fmax=5500,
                unique_filters=False,
            )
        )
    )

    ex.add_named_config(
        'chroma_clp',
        feature_extractor=dict(
            name='ChromaClp',
            params=dict(
                fps=10.022727272727273
            )
        )
    )

    ex.add_named_config(
        'perfect_chroma',
        feature_extractor=dict(
            name='PerfectChroma',
            params=dict(
                fps=10
            )
        )
    )


def create_extractor(config):
    return globals()[config['name']](**config['params'])
