import sys
from pathlib import Path
sys.path.append(str(Path(sys.argv[0]).absolute().parent.parent))

import time
import warnings
warnings.filterwarnings('ignore')

from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k

from ListenBrainz.utils import load_train_test, load_train_val

import numpy as np
import pandas as pd

def train(loss_type, epochs, n_components, alpha, precision, show=False):

    print('\nStarting')
    # Loading the data
    train_df, test_df = load_train_test()
    print('\nDataset loaded')

    # train_df = train_df.sample(n=10000)
    # test_df = test_df.sample(n=1000)

    # Preparing Train data
    popularity_train = train_df.groupby(['recording_mbid', 'user_id']).timestamp.nunique().reset_index()
    popularity_train.rename(columns={'timestamp': 'rating'}, inplace=True)
    mbid_mapping = popularity_train['recording_mbid'].drop_duplicates().reset_index(drop=True)
    mbid_mapping = mbid_mapping.reset_index().rename(columns={'index': 'music_id'})
    print('Mapping created')

    rated_train = pd.merge(popularity_train, mbid_mapping, on='recording_mbid', how='left')
    rated_train['user_id'] = rated_train['user_id'].astype(int)
    rated_train['music_id'] = rated_train['music_id'].astype(int)
    rated_train['rating'] = rated_train['rating'].astype(float)
    rated_train.drop('recording_mbid', axis=1, inplace=True)
    print('Ratings created')

    train_user_id = rated_train['user_id'].unique().astype('int64')
    # test_user_id = test_df['user_id'].unique().astype('int64')
    # test_user_id = np.intersect1d(test_user_id, train_user_id)

    # Preparing Validation data
    test_df = test_df[test_df['user_id'].isin(train_user_id)].reset_index(drop=True)
    popularity_val = test_df.groupby(['recording_mbid', 'user_id']).timestamp.nunique().reset_index()
    popularity_val.rename(columns={'timestamp': 'rating'}, inplace=True)
    rated_val = pd.merge(popularity_val, mbid_mapping, on='recording_mbid', how='left')
    max_index = mbid_mapping['music_id'].max() + 1
    rated_val['music_id'].fillna(max_index, inplace=True)
    rated_val['music_id'] = rated_val['music_id'].astype(int)
    print('Train and Test ratings created')

    data = Dataset()
    data.fit(train_user_id,
            rated_train['music_id'].unique().astype('int64'))

    (train_interactions, train_weights) = data.build_interactions([tuple(i) for i in rated_train.values.astype('int64')])
    print('Built Train Interactions')

    (test_interactions, test_weights) = data.build_interactions([tuple(i) for i in rated_val.values.astype('int64')])
    print('Built Test Interactions')

    print(f'Training LightFM with {loss_type} loss')
    start = time.time()
    model = LightFM(loss=loss_type, no_components=n_components, user_alpha=alpha)
    model = model.fit(interactions=train_interactions, sample_weight=train_weights, 
                    epochs=epochs, verbose=show)
    end = time.time()
    model_train_time = end-start
    print(f'Model training completed in {model_train_time} seconds')

    start = time.time()
    pred = model.predict(42, np.arrange(max_index))
    print(pred)
    end = time.time()
    print(f'Evaluation Time Corrected: {end-start}')

    # start = time.time()
    # train_precision = precision_at_k(model, train_interactions, k=precision).mean()
    # end = time.time()
    # print(f'Precision on Train: {train_precision}')
    # evaluate_time_train = end-start

    # start = time.time()
    # test_precision = precision_at_k(model, test_interactions, k=precision).mean()
    # end = time.time()
    # print(f'Precision on Test: {test_precision}')
    # evaluate_time_test = end-start

    # output = [f'\nLightFM Training Time: {model_train_time}', '\nParameters', 
    #       f'\n\tLoss: {loss_type}', f'\n\tEpochs: {epochs}',
    #       f'\n\tNum of Components: {n_components}', f'\n\tAlpha: {alpha}', f'\n\tPrecision at: {precision}',
    #       f'\mAP Train: {train_precision}', f'\nTime to evaluate Train: {evaluate_time_train}\n\n'
    #       f'\mAP Test: {test_precision}', f'\nTime to evaluate Test: {evaluate_time_test}\n\n']

    # # results = open('lightfm_results_final.txt', 'a')
    # # results.writelines(output)
    # # results.close()

    # print('\n\nFinal Result')
    # for line in output:
    #     print(line, end='')

if __name__ == '__main__':

    loss_type = 'warp'
    epochs = 10
    n_components = 10
    alpha = 0.1 
    precision = 100
    show = False

    parameters = [params.split('=') for params in sys.argv if params[:2]!='--' and params[0]=='-']
    parameters = {key[1:]: value for key, value in parameters}

    for key, value in parameters.items():
        if key in ['epoch', 'epochs']:
            epochs = int(value)
        if key in ['components', 'component', 'n_components']:
            n_components = int(value)
        if key in ['alpha']:
            alpha = float(value)
        if key in ['mAP', 'mean_average_precision', 'precision', 'precision_at', 'precision_at_k']:
            precision = int(value)
        if key in ['loss', 'model_type', 'loss_type']:
            loss_type = value
        if key in ['verbose', 'show']:
            show = bool(value)

    train(loss_type, epochs, n_components, alpha, precision, show)