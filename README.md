# chordrec

This is the code I use for my chord recognition experiments.

## Requirements & Installation

I assume a standard "scientific Python" environment with NumPy, SciPy, etc.
Additionally, the following libraries are required:

 - [Theano](https://github.com/Theano/Theano)
 - [Lasagne](https://github.com/Lasagne/Lasagne)
 - [dmgr](https://github.com/fdlm/dmgr)
 - [nn](https://github.com/fdlm/nn)
 - [Spaghetti](https://github.com/fdlm/Spaghetti)
 - [madmom](https://github.com/CPJKU/madmom)*
 - [librosa](https://github.com/librosa/librosa)*
 - [mir_eval](https://github.com/craffel/mir_eval)*
 - [pyyaml](https://bitbucket.org/xi/pyyaml)*
 - [sacred](https://github.com/IDSIA/sacred)*

Packages marked with a * can be installed using `pip`, the others are either
not available or recommended to be installed from source. If I missed any
dependency, please let me know.

Once you have all libraries installed, clone this repository and add its path
to the `$PYTHONPATH` environment variable.

## Data Setup

Different experiments might require different data set to be present (you can
find detailed information on the sites describing the experiments on my
[website](http://fdlm.github.io)). The directory structure for each dataset,
however, is the same.

Put all datasets into respective subdirectories under
`chordrec/experiments/data`. The datasets have to contain three types of data:
audio files in `.flac` format, corresponding chord annotations in lab format
with the file extension `.chords`, and the cross-validation split definitions.
Audio and annotation files can be organised on a directory structure, but do
not need to; the programs will look for any `.flac` and `.chord` files in all
directories recursively. However, the split definition
files must be in a `splits` sub-directory in each dataset directory (e.g.
`beatles/splits`). File names of audio and annotation files must correspond to
the names given in the split definition files.

The `data` directory including some example datasets should look like this,
The internal structures of the `queen`, `robbie_williams`, `rwc` and `zweieck`
directories following the one of the `beatles`:

```
experiments
 +-- data
      +-- beatles
           +-- *.flac
           +-- *.chords
           +-- splits
                +-- 8-fold_cv_album_distributed_*.fold
      +-- queen
      +-- robbie_williams
      +-- rwc
      +-- zweieck
```

Refer to the websites for each individual experiment for more information on
the data and how to obtain it.

## Experiments

The `experiments` sub-directory contains scripts and configurations to
reproduce the results of all my papers on chord recognition (plus some more).
Since neural networks are initialised randomly, and I usually do not save the
seed, the results might differ slightly from the ones in the papers.

 - `experiments/ismir2016`: Reproduces the final results for all features
   compared in the paper

   F. Korzeniowski and G. Widmer. ["Feature Learning for Chord Recognition: The
   Deep Chroma Extractor"](http://www.cp.jku.at/research/papers/Korzeniowski_ISMIR_2016.pdf). In *Proceedings of the 17th International Society
   for Music Information Retrieval Conference (ISMIR 2016)*,  New York, USA.

   See [here](http://fdlm.github.io/post/deepchroma) for more
   information on the model and the necessary data.

 - `experiments/madmom2016`: Configurations to train the chord recognition
   models of the [madmom](https://github.com/CPJKU/madmom) audio processing
   library.

 - `experiments/mlsp2016`: Reproduces the results of the chord recognition
   system presented in the following paper:

   F. Korzeniowski and G. Widmer. ["A Fully Convolutional Deep Auditory Model
   for Musical Chord Recognition"](http://www.cp.jku.at/research/papers/Korzeniowski_MLSP_2016.pdf)
   In *Proceedings of the IEEE International Workshop on Machine Learning for
   Signal Processing (MLSP 2016)*, Salerno, Italy, 2016.

   See [here](http://fdlm.github.io/post/mlsp2016) for more
   information on the model and the necessary data.
