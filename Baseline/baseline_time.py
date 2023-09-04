import sys
from pathlib import Path
sys.path.append(str(Path(sys.argv[0]).absolute().parent.parent))

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.mllib.evaluation import RankingMetrics

from ListenBrainz.utils import load_train_val, load_train_test
from ListenBrainz.metrics import meanAveragePrecision, PrecisionAtK

def main(spark, userID, data_type = "_ordered_small"):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''
    train_df, val_df = load_train_test(spark)

    print('Printing interactions inferred schema')
    train_df.printSchema()

    train_df = train_df.withColumn("timestamp_unix", col("timestamp").cast("timestamp").cast("long"))

    train_df.createOrReplaceTempView('train_interactions')
    val_df.createOrReplaceTempView('val_interactions')

    #####--------------YOUR CODE STARTS HERE--------------#####
    # train_df.show()
    best_beta = 0.0
    best_train_map = 0.0
    best_test_map = 0.0

    top_recommendations = [] # 0.0000001, 0.0000005, 0.000001,
    for decay_rate in [0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:

        popularity_train = train_df.groupBy("recording_mbid")\
                .agg((exp(-decay_rate*(max("timestamp_unix")- min("timestamp_unix")).cast("double")/86400)*countDistinct("user_id")).alias("popularity"))

        popularity_train = popularity_train.orderBy(col("popularity").desc()).limit(100)
        # popularity_train.show()
        top_train_100 = [row["recording_mbid"] for row in popularity_train.select("recording_mbid").collect()]

        print("\nBeta = {}".format(decay_rate))
        train_map_value = PrecisionAtK(spark, train_df, top_train_100, 10)
        print("Train Precision at 10 Value ={}".format(train_map_value))
        train_map_value = PrecisionAtK(spark, train_df, top_train_100, 50)
        print("Train Precision at 50 Value ={}".format(train_map_value))
        train_map_value = PrecisionAtK(spark, train_df, top_train_100, 100)
        print("Train Precision at 100 Value ={}".format(train_map_value))

        test_map_value = PrecisionAtK(spark, val_df, top_train_100, 10)
        print("Validation Precision at 10 Value ={}".format(test_map_value))
        test_map_value = PrecisionAtK(spark, val_df, top_train_100, 50)
        print("Validation Precision at 50 Value ={}".format(test_map_value))
        test_map_value = PrecisionAtK(spark, val_df, top_train_100, 100)
        print("Validation Precision at 100 Value ={}".format(test_map_value))

        if test_map_value>best_test_map:
            best_beta = decay_rate
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