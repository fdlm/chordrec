from __future__ import print_function

import os
import numpy as np
import yaml

from nn.utils import Colors

import data
import dmgr
import features
import nn
import targets
import test
from experiment import TempDir, create_optimiser, setup
from models import chroma_dnn


def compute_chroma(process_fn, agg_dataset, dest_dir, batch_size,
                   extension='.features.npy'):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    chroma_files = []

    for ds_idx in range(agg_dataset.n_datasources):
        ds = agg_dataset.datasource(ds_idx)

        chromas = []

        for data, _ in dmgr.iterators.iterate_batches(ds, batch_size,
                                                      randomise=False,
                                                      expand=False):
            chromas.append(process_fn(data))

        chromas = np.concatenate(chromas)
        chroma_file = os.path.join(dest_dir, ds.name + extension)
        np.save(chroma_file, chromas)
        chroma_files.append(chroma_file)

    return chroma_files


# Initialise Sacred experiment
ex = setup('Deep Chroma Extractor')


# Standard config
@ex.config
def _cfg():
    observations = 'results'
    feature_extractor = None
    target = None
    chroma_network = None
    optimiser = None
    training = None
    regularisation = None
    testing = None
    augmentation = None


# add models
chroma_dnn.add_sacred_config(ex)


@ex.automain
def main(datasource, feature_extractor, target, chroma_network,
         optimiser, training, regularisation, augmentation, testing):

    err = False
    if chroma_network is None:
        print(Colors.red('ERROR: Specify a chroma extractor!'))
        err = True
    if feature_extractor is None:
        print(Colors.red('ERROR: Specify a feature extractor!'))
        err = True
    if target is None:
        print(Colors.red('ERROR: Specify a target!'))
        err = True
    if chroma_network is None:
        print(Colors.red('ERROR: Specify a chroma extractor!'))
        err = True
    if err:
        return 1

    # intermediate target is always chroma vectors
    target_chroma = targets.ChromaTarget(
        feature_extractor['params']['fps'])

    target_chords = targets.create_target(
        feature_extractor['params']['fps'],
        target
    )

    if not isinstance(datasource['test_fold'], list):
        datasource['test_fold'] = [datasource['test_fold']]

    if not isinstance(datasource['val_fold'], list):
        datasource['val_fold'] = [datasource['val_fold']]

        # if no validation folds are specified, always use the
        # 'None' and determine validation fold automatically
        if datasource['val_fold'][0] is None:
            datasource['val_fold'] *= len(datasource['test_fold'])

    if len(datasource['test_fold']) != len(datasource['val_fold']):
        print(Colors.red('ERROR: Need same number of validation and '
                         'test folds'))
        return 1

    all_pred_files = []
    all_gt_files = []

    print(Colors.magenta('\nStarting experiment ' + ex.observers[0].hash()))

    with TempDir() as exp_dir:
        for test_fold, val_fold in zip(datasource['test_fold'],
                                       datasource['val_fold']):
            print('')
            print(Colors.yellow(
                '=' * 20 + ' FOLD {} '.format(test_fold) + '=' * 20))
            # Load data sets
            print(Colors.red('\nLoading data...\n'))

            feature_ext = features.create_extractor(feature_extractor,
                                                    test_fold)
            train_set, val_set, test_set, gt_files = data.create_datasources(
                dataset_names=datasource['datasets'],
                preprocessors=datasource['preprocessors'],
                compute_features=feature_ext,
                compute_targets=target_chroma,
                context_size=datasource['context_size'],
                test_fold=test_fold,
                val_fold=val_fold,
                cached=datasource['cached']
            )

            if testing['test_on_val']:
                test_set = val_set

            print(Colors.blue('Train Set:'))
            print('\t', train_set)

            print(Colors.blue('Validation Set:'))
            print('\t', val_set)

            print(Colors.blue('Test Set:'))
            print('\t', test_set)
            print('')

            # build network
            print(Colors.red('Building network...\n'))

            model_type = globals()[chroma_network['model']['type']]
            mdl = model_type.build_model(in_shape=train_set.dshape,
                                         out_size_chroma=train_set.tshape[0],
                                         out_size=target_chords.num_classes,
                                         model=chroma_network['model'])

            chroma_neural_net = mdl['chroma_network']
            chord_neural_net = mdl['chord_network']
            input_var = mdl['input_var']
            chroma_target_var = mdl['chroma_target_var']
            chord_target_var = mdl['chord_target_var']
            chroma_loss_fn = mdl['chroma_loss_fn']
            chord_loss_fn = mdl['chord_loss_fn']

            chroma_opt, chroma_lrs = create_optimiser(chroma_network['optimiser'])
            chord_opt, chord_lrs = create_optimiser(optimiser)

            chroma_train_fn = nn.compile_train_fn(
                chroma_neural_net, input_var, chroma_target_var,
                loss_fn=chroma_loss_fn, opt_fn=chroma_opt,
                **chroma_network['regularisation']
            )

            chroma_test_fn = nn.compile_test_func(
                chroma_neural_net, input_var, chroma_target_var,
                loss_fn=chroma_loss_fn,
                **chroma_network['regularisation']
            )

            chroma_process_fn = nn.compile_process_func(
                chroma_neural_net, input_var
            )

            chord_train_fn = nn.compile_train_fn(
                chord_neural_net, input_var, chord_target_var,
                loss_fn=chord_loss_fn, opt_fn=chord_opt, tags={'chord': True},
                **regularisation
            )

            chord_test_fn = nn.compile_test_func(
                chord_neural_net, input_var, chord_target_var,
                loss_fn=chord_loss_fn, tags={'chord': True},
                **regularisation
            )

            chord_process_fn = nn.compile_process_func(
                chord_neural_net, input_var
            )

            print(Colors.blue('Chroma Network:'))
            print(nn.to_string(chroma_neural_net))
            print('')

            print(Colors.blue('Chords Network:'))
            print(nn.to_string(chord_neural_net))
            print('')

            print(Colors.red('Starting training chroma network...\n'))

            chroma_training = chroma_network['training']
            chroma_train_batches, chroma_validation_batches = \
                model_type.create_iterators(train_set, val_set,
                                            chroma_training, augmentation)
            crm_train_losses, crm_val_losses, _, crm_val_accs = nn.train(
                network=chroma_neural_net,
                train_fn=chroma_train_fn, train_batches=chroma_train_batches,
                test_fn=chroma_test_fn,
                validation_batches=chroma_validation_batches,
                threads=10, callbacks=[chroma_lrs] if chroma_lrs else [],
                num_epochs=chroma_training['num_epochs'],
                early_stop=chroma_training['early_stop'],
                early_stop_acc=chroma_training['early_stop_acc'],
                acc_func=nn.nn.elemwise_acc
            )

            # we need to create a new dataset with a new target (chords)
            del train_set
            del val_set
            del test_set
            del gt_files

            train_set, val_set, test_set, gt_files = data.create_datasources(
                dataset_names=datasource['datasets'],
                preprocessors=datasource['preprocessors'],
                compute_features=feature_ext,
                compute_targets=target_chords,
                context_size=datasource['context_size'],
                test_fold=test_fold,
                val_fold=val_fold,
                cached=datasource['cached']
            )

            if testing['test_on_val']:
                test_set = val_set

            print(Colors.blue('Train Set:'))
            print('\t', train_set)

            print(Colors.blue('Validation Set:'))
            print('\t', val_set)

            print(Colors.blue('Test Set:'))
            print('\t', test_set)
            print('')

            print(Colors.red('Starting training chord network...\n'))

            chord_train_batches, chord_validation_batches = \
                model_type.create_iterators(train_set, val_set, training,
                                            augmentation)

            crd_train_losses, crd_val_losses, _, crd_val_accs = nn.train(
                network=chord_neural_net,
                train_fn=chord_train_fn, train_batches=chord_train_batches,
                test_fn=chord_test_fn,
                validation_batches=chord_validation_batches,
                threads=10, callbacks=[chord_lrs] if chord_lrs else [],
                num_epochs=training['num_epochs'],
                early_stop=training['early_stop'],
                early_stop_acc=training['early_stop_acc'],
            )

            print(Colors.red('\nStarting testing...\n'))

            param_file = os.path.join(
                exp_dir, 'params_fold_{}.pkl'.format(test_fold))
            nn.save_params(chord_neural_net, param_file)
            ex.add_artifact(param_file)

            pred_files = test.compute_labeling(
                chord_process_fn, target_chords, test_set, dest_dir=exp_dir,
                use_mask=False, batch_size=testing['batch_size']
            )

            # compute chroma vectors for the test set
            # TODO: replace this with experiment.compute_features
            for cf in compute_chroma(chroma_process_fn, test_set,
                                     batch_size=training['batch_size'],
                                     dest_dir=exp_dir):
                ex.add_artifact(cf)

            test_gt_files = dmgr.files.match_files(
                pred_files, test.PREDICTION_EXT, gt_files, data.GT_EXT
            )

            all_pred_files += pred_files
            all_gt_files += test_gt_files

            print(Colors.blue('Results:'))
            scores = test.compute_average_scores(test_gt_files, pred_files)
            test.print_scores(scores)
            result_file = os.path.join(
                exp_dir, 'results_fold_{}.yaml'.format(test_fold))
            yaml.dump(dict(scores=scores,
                           chord_train_losses=map(float, crd_train_losses),
                           chord_val_losses=map(float, crd_val_losses),
                           chord_val_accs=map(float, crd_val_accs),
                           chroma_train_losses=map(float, crm_train_losses),
                           chroma_val_losses=map(float, crm_val_losses),
                           chroma_val_accs=map(float, crm_val_accs)),
                      open(result_file, 'w'))
            ex.add_artifact(result_file)

            # close all files
            del train_set
            del val_set
            del test_set
            del gt_files

        # if there is something to aggregate
        if len(datasource['test_fold']) > 1:
            print(Colors.yellow('\nAggregated Results:\n'))
            scores = test.compute_average_scores(all_gt_files, all_pred_files)
            test.print_scores(scores)
            result_file = os.path.join(exp_dir, 'results.yaml')
            yaml.dump(dict(scores=scores), open(result_file, 'w'))
            ex.add_artifact(result_file)

        for pf in all_pred_files:
            ex.add_artifact(pf)

    print(Colors.magenta('Stopping experiment ' + ex.observers[0].hash()))
