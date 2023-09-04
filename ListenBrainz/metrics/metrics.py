from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.mllib.evaluation import RankingMetrics

def meanAveragePrecision(spark, val_df, top_train_100):

    listed_val_df = val_df.groupBy(col('user_id')).agg(collect_list(col('recording_mbid')).alias('actual_items'))
    user_recs = spark.createDataFrame([(user_id, top_train_100) for user_id in val_df.select('user_id').distinct().rdd.map(lambda row: row[0]).collect()], ['user_id', 'recommended_items'])
    val_data_with_recs = listed_val_df.join(user_recs, 'user_id', 'inner')
    user_recs_and_actual = val_data_with_recs.select('user_id', 'recommended_items', 'actual_items').rdd.map(lambda row: (row[1], row[2]))

    metrics = RankingMetrics(user_recs_and_actual)
    map_value = metrics.meanAveragePrecision
    return map_value

def PrecisionAtK(spark, val_df, top_train_100, k):

    listed_val_df = val_df.groupBy(col('user_id')).agg(collect_list(col('recording_mbid')).alias('actual_items'))
    user_recs = spark.createDataFrame([(user_id, top_train_100) for user_id in val_df.select('user_id').distinct().rdd.map(lambda row: row[0]).collect()], ['user_id', 'recommended_items'])
    val_data_with_recs = listed_val_df.join(user_recs, 'user_id', 'inner')
    user_recs_and_actual = val_data_with_recs.select('user_id', 'recommended_items', 'actual_items').rdd.map(lambda row: (row[1], row[2]))

    metrics = RankingMetrics(user_recs_and_actual)
    patk = metrics.precisionAt(k)
    return patk