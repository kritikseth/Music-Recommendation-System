import sys
from pathlib import Path
sys.path.append(str(Path(sys.argv[0]).absolute().parent.parent))

import pandas as pd
from ListenBrainz.utils import (
    get_greene_path, get_greene_user_path
)

def split():

    file_type, file_size = 'ordered', ''
    train_size = 0.7

    if '--ordered' in sys.argv:
        file_type = 'ordered'
    elif '--random' in sys.argv:
        file_type = 'random'
    
    if '--small' in sys.argv:
        file_size = '_small'

    train_size = 0.7

    parameters = [params.split('=') for params in sys.argv if params[:2]!='--' and params[0]=='-']
    parameters = {key[1:]: value for key, value in parameters}

    for key, value in parameters.items():
        if key in ['train', 'train_size']:
            train_size = float(value)
        if key in ['test', 'test_size']:
            train_size = 1 - float(value)

    interactions_filename = f'interactions_train{file_size}'
    tracks_filename = f'tracks_train{file_size}'
    
    interactions = pd.read_parquet(get_greene_path(interactions_filename))
    tracks = pd.read_parquet(get_greene_path(tracks_filename))

    music_data = interactions.merge(tracks, on='recording_msid', how='left')
    music_data.reset_index(inplace=True)
    music_data['recording_mbid'].fillna(music_data['recording_msid'], inplace=True)
    music_data = music_data[['user_id', 'recording_mbid', 'timestamp']]

    # Print the total rows before dropping NA
    print(f'Total Rows: {len(music_data)}')

    # Drop rows with NA values in 'user_id' or 'recording_mbid'
    music_data = music_data.dropna(subset=['user_id', 'recording_mbid'])

    # Print the total rows after dropping NA
    print(f'Total Rows After Dropping NA: {len(music_data)}')

    # Add interaction_number column using groupby and cumcount
    music_data['interaction_number'] = music_data.groupby('user_id').cumcount()

    # Get the split index as 70% of the maximum interaction_number per user
    split_index = music_data.groupby('user_id')['interaction_number'].max() * train_size
    
    # Split the data into train and val based on split index
    train = music_data[music_data['interaction_number'] < music_data['user_id'].map(split_index)]
    val = music_data[music_data['interaction_number'] >= music_data['user_id'].map(split_index)]

    # Print the total rows in train and val data
    print(f'Total Rows in Train Data: {len(train)}')
    print(f'Total Rows in Val Data: {len(val)}')
    
    train_filename = f'train_{file_type}{file_size}'
    val_filename = f'val_{file_type}{file_size}'
    
    train.to_parquet(get_greene_user_path(train_filename))
    val.to_parquet(get_greene_user_path(val_filename))

    print('Write Complete!')

if __name__ == '__main__':

    split()