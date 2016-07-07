# Running the experiment for the MLSP 2016 paper

The experiment consists of two steps. First, we train the feature extraction
CNN. Second, we train the conditional random field that decodes chord
sequences.

## CNN feature extractor

To train the convnet, simply run

    $ python -m chordrec.classify with convnet.yaml

and note the experiment id (`<cn_expid>`).

## CRF chord decoder

First, create the CRF parameter initialisation files for each fold. We
will save those into a subdirectory `crf_init_params`:

    $ ./create_crf_init_params.py results/<cn_expid>/artifacts crf_init_params

and train the CRF for chord sequence decoding:

    $ python -m chordrec.classify with crf.yaml \
                feature_extractor.params.name='../../results/<cn_expid>/artifacts/features_fold_{fold}' \
                training.init_file='crf_init_params/crf_init_params_{}.pkl'
