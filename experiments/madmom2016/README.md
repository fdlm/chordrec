# Training Models for the `madmom` Audio Processing Framework

The following text will guide you through the process of training
chord-recognition related models of `madmom`.

## Deep Chroma Extractor

To train the Deep Chroma Extractor model, simply run

    $ python -m chordrec.chroma with deep_chroma.yaml

and note the experiment ID (`<dc_expid>`). After training finished, you can
convert the learned model to a madmom compatible model by running

    $ ./create_madmom_deep_chroma_model.py results/<dc_expid>/artifacts/params_fold_None.pkl \
                                           chroma_dnn.pkl

This will create a file "chords_dnn.pkl" which contains the madmom
neural network model.

## Deep Chroma Chord Recogniser

Before training the deep chroma chord recogniser, make sure to train the
deep chroma extractor and note its experiment ID. The trained chord recogniser
will work best with this chroma extractor.

To train the Deep Chroma Chord Recogniser, run

    $ python -m chordrec.classify with crf_chord_rec.yaml \
                feature_extractor.params.name='../../results/<dc_expid>/artifacts'

and note the experiment ID (`<dccr_expid>`). Then, convert the learned model
to the a madmom compatible one using

    $ ./create_madmom_crf_model.py results/<dccr_expid>/params_fold_None.pkl \
                                   chords_dccrf.pkl

This will create a file "chords_dccrf.pkl" which contains the madmom CRF model
for chord recognition.

## ConvNet Chord Recogniser

The ConvNet Chord Recogniser consists of a) the feature extraction ConvNet
and b) a CRF for decoding the chord sequence. First, you need to train
the ConvNet:

    $ python -m chordrec.classify with chord_feature_convnet.yaml

Note the experiment id (`<cn_expid>`). Then, create the parameter
initialisation file for the CRF,

    $ ./create_crf_init_params.py results/<cn_expid>/artifacts/params_fold_None.pkl \
                                  crf_init_params.pkl

and train the CRF for chord sequence decoding:

    $ python -m chordrec.classify with crf_chord_rec.yaml \
                feature_extractor.params.name='../../results/<cn_expid>/artifacts/features_fold_None' \
                training.init_file='crf_init_params.pkl'

Also note the corresponding experiment id (`<cncr_expid>`). Then, convert the
learned models to madmom models:

    $ ./create_madmom_convnet_model.py results/<cn_expid>/artifacts/params_fold_None.pkl \
                                       chords_cnnfeat.pkl
    $ ./create_madmom_crf_model.py results/<cncr_expid>/artifacts/params_fold_None.pkl \
                                   chords_cnncrf.pkl

This will create two files (`chords_cnnfeat.pkl` and `chords_cnncrf.pkl`) which
contain the CNN feature extraction model and the CRF chord recognition model
respectively.