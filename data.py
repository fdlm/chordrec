from operator import eq
import os
import dmgr

DATA_DIR = 'data'
CACHE_DIR = 'feature_cache'
SRC_EXT = '.flac'
GT_EXT = '.chords'


def combine_files(*args):
    """
    Combines file dictionaries as returned by the methods of Dataset.
    :param args: file dictionaries
    :return:     combined file dictionaries
    """
    if len(args) < 1:
        raise ValueError('Pass at least one argument!')

    # make sure all elements contain the same number of splits
    if not reduce(eq, map(len, args)):
        raise ValueError('Arguments must contain the same number of splits!')

    combined = [{'feat': [], 'targ': []} for _ in range(len(args[0]))]

    for fs in args:
        for s in range(len(combined)):
            for t in combined[s]:
                combined[s][t] += fs[s][t]

    return combined


DATASET_DEFS = {
    'beatles': {
        'data_dir': 'beatles',
        'split_filename': '8-fold_cv_album_distributed_{}.fold'
    },
    'queen': {
        'data_dir': 'queen',
        'split_filename': '8-fold_cv_random_{}.fold'
    },
    'zweieck': {
        'data_dir': 'zweieck',
        'split_filename': '8-fold_cv_random_{}.fold'
    },
    'robbie_williams': {
        'data_dir': 'robbie_williams',
        'split_filename': '8-fold_cv_random_{}.fold'
    },
    'rwc': {
        'data_dir': 'rwc',
        'split_filename': '8-fold_cv_random_{}.fold'
    },
    'billboard': {
        'data_dir': os.path.join('mcgill-billboard', 'unique'),
        'split_filename': '8-fold_cv_random_{}.fold'
    }
}


def load_dataset(name, data_dir, feature_cache_dir,
                 compute_features, compute_targets):

    assert name in DATASET_DEFS.keys(), 'Unknown dataset {}'.format(name)

    data_dir = os.path.join(data_dir, DATASET_DEFS[name]['data_dir'])
    split_filename = os.path.join(data_dir, 'splits',
                                  DATASET_DEFS[name]['split_filename'])

    return dmgr.Dataset(
        data_dir,
        os.path.join(feature_cache_dir, name),
        [split_filename.format(f) for f in range(8)],
        source_ext=SRC_EXT,
        gt_ext=GT_EXT,
        compute_features=compute_features,
        compute_targets=compute_targets,
    )


def create_preprocessors(preproc_defs):
    preprocessors = []
    for pp in preproc_defs:
        preprocessors.append(
            getattr(dmgr.preprocessing, pp['name'])(**pp['params']))
    return preprocessors


def create_datasources(dataset_names, preprocessors,
                       compute_features, compute_targets, context_size,
                       data_dir=DATA_DIR, feature_cache_dir=CACHE_DIR,
                       test_fold=0, val_fold=None,
                       **kwargs):

    val_fold = val_fold if val_fold is not None else test_fold - 1
    preprocessors = create_preprocessors(preprocessors)

    if context_size > 0:
        data_source_type = dmgr.datasources.ContextDataSource
        kwargs['context_size'] = context_size
    else:
        data_source_type = dmgr.datasources.DataSource

    # load all datasets
    datasets = [load_dataset(name, data_dir, feature_cache_dir,
                             compute_features, compute_targets)
                for name in dataset_names]

    # uses fold 0 for validation, fold 1 for test, rest for training
    train, val, test = dmgr.datasources.get_datasources(
        combine_files(*[ds.fold_split(val_fold, test_fold)
                        for ds in datasets]),
        preprocessors=preprocessors, data_source_type=data_source_type,
        **kwargs
    )

    return train, val, test, sum((ds.gt_files for ds in datasets), [])


def add_sacred_config(ex):
    ex.add_config(
        datasource=dict(
            datasets=['beatles', 'queen', 'zweieck', 'robbie_williams', 'rwc'],
            context_size=0,
            preprocessors=[],
            # fold 6 overestimates the score, but has highest correlation
            # with the total score
            test_fold=6,
            val_fold=None,
            cached=True
        )
    )
