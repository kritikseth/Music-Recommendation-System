import sys
from pathlib import Path
sys.path.append(str(Path(sys.argv[0]).absolute().parent.parent))

import os
import time

from pyspark.sql import SparkSession,Window
from pyspark.ml.evaluation import RegressionEvaluator,RankingEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from pyspark.sql.functions import *

from ListenBrainz.utils.constants import GREENE_DATA
from ListenBrainz.utils import load_train_val

def latent_factor(spark, userID, train_df, test_df, data_size):
    train_df.createOrReplaceTempView('train_interactions')
    test_df.createOrReplaceTempView('test_interactions')
    
    print("\n\nCreating Train Data\n\n")
    # Preparing mapper
    mbid_mapping = spark.sql('SELECT DISTINCT(recording_mbid), ROW_NUMBER() OVER (ORDER BY recording_mbid) AS music_id FROM train_interactions GROUP BY recording_mbid')
    mbid_mapping.createOrReplaceTempView('mbid_mapping')

    # Preparing Train data
    popularity_train = spark.sql('SELECT recording_mbid, user_id, COUNT(DISTINCT(timestamp)) as rating FROM train_interactions GROUP BY recording_mbid, user_id')
    popularity_train.createOrReplaceTempView('popularity_train')
    rated_train = spark.sql('SELECT p.recording_mbid, m.music_id, p.user_id, p.rating FROM popularity_train as p LEFT JOIN mbid_mapping as m ON p.recording_mbid = m.recording_mbid')
    rated_train = rated_train.withColumn('user_id', col('user_id').cast('integer')).withColumn('music_id', col('music_id').cast('integer')).withColumn('rating', col('rating').cast('float')).drop('recording_mbid')
    max_index = mbid_mapping.agg(max('music_id')).first()[0] + 1

    print("\n\nTrain Data Created\n\n")
    print("\n\nTotal Unique Songs in Train:", max_index, "\n\n")

    # Add hyperparameters and their respective values to param_grid
    best_rank = 35
    best_regParam = 0.2

    # Train
    als = ALS(
            maxIter=5,
            rank=best_rank,
            regParam=best_regParam,
            userCol="user_id",
            itemCol="music_id",
            ratingCol="rating",
            nonnegative = True,
            implicitPrefs = False,
            coldStartStrategy="drop"
            )

    # Define evaluator as RMSE and print length of evaluator
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    print("\n\nModel Trainig Start\n\n")
    start = time.time()
    model = als.fit(rated_train)
    end = time.time()
    elapsed_time = end - start
    print("\n\nModel Trainig End\n\n")
    print("\n\nTime to fit model:", elapsed_time, "\n\n")

    # Preparing Validation data
    print("\n\nCreating Validation Data\n\n")
    popularity_test = spark.sql('SELECT recording_mbid, user_id, COUNT(DISTINCT(timestamp)) as rating FROM test_interactions GROUP BY recording_mbid, user_id')
    popularity_test.createOrReplaceTempView('popularity_test')
    rated_test = spark.sql('SELECT p.recording_mbid, m.music_id, p.user_id, p.rating FROM popularity_test as p LEFT JOIN mbid_mapping as m ON p.recording_mbid = m.recording_mbid')
    rated_test = rated_test.na.fill(max_index, subset=['music_id'])
    rated_test= rated_test.withColumn('user_id', col('user_id').cast('integer')).withColumn('music_id', col('music_id').cast('integer')).withColumn('rating', col('rating').cast('float')).drop('recording_mbid')
    print("\n\Validation Data Created\n\n")
    
    # Validation
    print("\n\nPredicting\n\n")
    start = time.time()
    test_pred = model.transform(rated_test)    
    end = time.time()
    evaluation_time = end - start

    rmse = evaluator.evaluate(test_pred)

    print("\n\nCreating Prediction Data\n\n")
    window = Window.partitionBy(test_pred['user_id']).orderBy(test_pred['prediction'].desc())  
    test_pred = test_pred.withColumn('rank', rank().over(window)).filter(col('rank') <= 100).groupby("user_id").agg(collect_list(test_pred['music_id'].cast('double')).alias('pred_music'))
    window = Window.partitionBy(rated_test['user_id']).orderBy(rated_test['rating'].desc())  
    df_mov = rated_test.withColumn('rank', rank().over(window)).filter(col('rank') <= 100).groupby("user_id").agg(collect_list(rated_test['music_id'].cast('double')).alias('music'))
    
    test_pred = test_pred.join(df_mov, test_pred.user_id==df_mov.user_id).drop('user_id')

    # print("\n\nPrediction Data:")
    # test_pred.show()

    print("\n\nCalculating Metrics\n\n")
    metrics = ['meanAveragePrecision','meanAveragePrecisionAtK','precisionAtK','ndcgAtK','recallAtK']
    metricsDict = {
        'rmse':rmse
    }
    for metric in metrics:
        rEvaluator = RankingEvaluator(predictionCol='pred_music', labelCol='music', metricName=metric, k=best_rank)
        metricsDict[metric] = rEvaluator.evaluate(test_pred)

    print("\n\nCalculation Complete\n\n")

    print("Rank                         :", best_rank)
    print("Regularization Parameter     :", best_regParam)
    print("Time to fit model            :", elapsed_time)
    print("Time to evaluate model       :", evaluation_time)
    print("Mean Average Precision       :", metricsDict["meanAveragePrecision"])
    print(metricsDict)
                
    print("\n\nSaving Model\n\n")
    mbid_mapping.write.parquet(f'hdfs:/user/{userID}/mapper{data_size}.parquet')
    model.save("./models")

if __name__ == '__main__':

    spark = SparkSession.builder.appName('Latent Factor Model').getOrCreate()
    
    file_type, file_size = 'ordered', ''

    if '--ordered' in sys.argv:
        file_type = 'ordered'
        if '--small' in sys.argv:
            file_size = '_small'

    elif '--random' in sys.argv:
        file_type = 'random'
        if '--small' in sys.argv:
            file_size = '_small'
    
    train_df, test_df = load_train_val(spark)
    userID = os.environ['USER']
    if len(file_size):
        file_size = "_"+file_size
    latent_factor(spark, userID, train_df, test_df, file_size)