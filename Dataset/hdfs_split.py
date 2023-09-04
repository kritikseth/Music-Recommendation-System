import sys
from pathlib import Path
sys.path.append(str(Path(sys.argv[0]).absolute().parent.parent))

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import time


def main(spark, userID, filename = 'interactions_train_small'):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''

    get_path = lambda x: f'/user/bm106_nyu_edu/1004-project-2023/{x}.parquet'
    interactions = spark.read.parquet(get_path(filename))

    print('Printing interactions inferred schema')
    interactions.printSchema()

    #####--------------YOUR CODE STARTS HERE--------------#####

    if filename[-6:] == "_small":
        train_file_path = f'hdfs:/user/{userID}/interactions_train_ordered_small.parquet'
        val_file_path = f'hdfs:/user/{userID}/interactions_val_ordered_small.parquet'
        track_file_path = 'tracks_train_small'
    else:
        train_file_path = f'hdfs:/user/{userID}/interactions_train_ordered.parquet'
        val_file_path = f'hdfs:/user/{userID}/interactions_val_ordered.parquet'
        track_file_path = 'tracks_train'
    
    tracks = spark.read.parquet(get_path(track_file_path))
    music_data = interactions.join(tracks, 'recording_msid', 'left')
    music_data = music_data.withColumn('recording_mbid', coalesce('recording_mbid', 'recording_msid'))
    interactions = music_data.select('user_id', 'recording_mbid', 'timestamp')

    print("Total Rows:", interactions.count())
    interactions = interactions.na.drop(subset=['user_id', 'recording_mbid'])

    print("Total Rows After Dropping NA:", interactions.count())

    print("\n\n\nRaw Data")
    start_time = time.time()
    interactions.show()

    interactions.createOrReplaceTempView('interactions')

    interactions_with_rowid = spark.sql('SELECT user_id, recording_mbid, timestamp, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY user_id, timestamp) - 1 AS interaction_number FROM interactions')
    interactions_with_rowid.show()
    interactions_with_rowid.createOrReplaceTempView('interactions_with_rowid')

    print("Total Rows in Indexed Data:", interactions_with_rowid.count())

    split_index = spark.sql('SELECT user_id,  MAX(interaction_number)*0.7 AS split_index FROM interactions_with_rowid GROUP BY user_id')
    split_index.createOrReplaceTempView('split_index')
    split_index.show()

    train = spark.sql('SELECT i.user_id, i.recording_mbid, i.timestamp FROM interactions_with_rowid as i LEFT JOIN split_index as s ON i.user_id = s.user_id WHERE i.interaction_number < s.split_index')
    print("Total Rows in Train Data:", train.count())

    val = spark.sql('SELECT i.user_id, i.recording_mbid, i.timestamp FROM interactions_with_rowid as i LEFT JOIN split_index as s ON i.user_id = s.user_id WHERE i.interaction_number >= s.split_index')
    print("Total Rows in Val Data:", val.count())

    train.write.parquet(f'hdfs:/user/{userID}/train_ordered.parquet')
    val.write.parquet(f'hdfs:/user/{userID}/val_ordered.parquet')



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('train_val_split_ordered').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']
    file_path = sys.argv[1]

    # Call our main routine
    main(spark, userID, file_path)