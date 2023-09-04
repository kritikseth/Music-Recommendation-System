import sys
from pathlib import Path
sys.path.append(str(Path(sys.argv[0]).absolute().parent.parent))

import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.mllib.evaluation import RankingMetrics

from ListenBrainz.utils import load_train_test
import sys

def main(spark, userID, data_type = "_ordered_small"):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''
    # interactions_train_file_name = "interactions_train" + data_type
    # interactions_val_file_name = "interactions_val"+ data_type
    # get_path = lambda x: f'hdfs:/user/{userID}/{x}.parquet'
    # train_df = spark.read.parquet(get_path(interactions_train_file_name))
    # val_df = spark.read.parquet(get_path(interactions_val_file_name))

    train_df, val_df = load_train_test(spark)

    print('Printing interactions inferred schema')
    train_df.printSchema()

    train_df.createOrReplaceTempView('train_interactions')
    val_df.createOrReplaceTempView('val_interactions')

    #####--------------YOUR CODE STARTS HERE--------------#####
    # train_df.show()
    best_beta = 0.0
    best_train_map = 0.0
    best_test_map = 0.0
    top_recommendations = []
    for beta in [100, 1000, 5000, 10000, 50000, 100000]:
        popularity_train = spark.sql("SELECT recording_mbid, COUNT(user_id) as total_plays, COUNT(DISTINCT(user_id)) as total_users, COUNT(user_id)/(COUNT(DISTINCT(user_id)) + {}) as average_plays FROM train_interactions GROUP BY recording_mbid ORDER BY average_plays DESC LIMIT 100".format(beta))
        # popularity_train.show()
        top_train_100 = [row["recording_mbid"] for row in popularity_train.select("recording_mbid").collect()]

        print("\nBeta = {}".format(beta))
        train_map_value = get_MAP(spark, train_df, top_train_100)
        print("Train MAP Value ={}".format(train_map_value))
        test_map_value = get_MAP(spark, val_df, top_train_100)
        print("Validation MAP Value ={}".format(test_map_value))

        if test_map_value>best_test_map:
            best_beta = beta
            best_train_map = train_map_value
            best_test_map = test_map_value
            top_recommendations = top_train_100

    print("\n\n\nBest Beta = {}".format(best_beta))
    print("Best Test MAP = {}".format(best_test_map))
    print("Train MAP = {}".format(best_train_map))
    print("Best Recommendations = {}".format(top_recommendations))
    print()



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']
    # data_type = sys.argv[1]

    # Call our main routine
    main(spark, userID)