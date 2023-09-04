import sys
from pathlib import Path
sys.path.append(str(Path(sys.argv[0]).absolute().parent.parent))

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from ListenBrainz.utils import (
    get_greene_path,
    get_greene_user_path,
    get_hdfs_path,
    get_hdfs_user_path,
    NETID
)

def merge_greene():
    
    interactions = pd.read_parquet(get_greene_path('interactions_train'))
    tracks = pd.read_parquet(get_greene_path('tracks_train'))

    music_data = interactions.merge(tracks, on='recording_msid', how='left')
    music_data.reset_index(inplace=True)
    music_data['recording_mbid'].fillna(music_data['recording_msid'], inplace=True)
    music_data = music_data[['user_id', 'recording_mbid', 'timestamp']]
    
    music_data.to_parquet(get_greene_user_path('train'))
    print('Write Complete!')

def merge_hdfs(spark):

    interactions = spark.read.parquet(get_hdfs_path('interactions_train'))
    tracks = spark.read.parquet(get_hdfs_path('tracks_train'))

    music_data = interactions.join(tracks, 'recording_msid', 'left')
    music_data = music_data.withColumn('recording_mbid', coalesce('recording_mbid', 'recording_msid'))
    music_data = music_data.na.drop(subset=['user_id', 'recording_mbid'])
    music_data = music_data.select('user_id', 'recording_mbid', 'timestamp')

    music_data.write.parquet(get_hdfs_user_path('train'))
    print('Write Complete!')

if __name__ == '__main__':

    if NETID.endswith('_nyu_edu'):
        spark = SparkSession.builder.appName('Create Train Data').getOrCreate()
        merge_hdfs(spark)
    else:
        merge_greene()