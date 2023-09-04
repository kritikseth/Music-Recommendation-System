import sys
from pathlib import Path
sys.path.append(str(Path(sys.argv[0]).absolute().parent.parent))

from ListenBrainz.utils import load_train_val
from ListenBrainz.metrics import meanAveragePrecision

from pyspark.sql import SparkSession
from pyspark.sql.functions import *


def baseline_1(spark, train_df, val_df):

    train_df.createOrReplaceTempView('train_interactions')
    val_df.createOrReplaceTempView('val_interactions')

    best_beta = 0.0
    best_train_map, best_test_map = 0.0, 0.0
    top_recommendations, beta_values = [], [100, 1000, 5000, 10000, 50000, 100000]

    for beta in beta_values:

        popularity_train = spark.sql('SELECT recording_mbid, COUNT(user_id) as total_plays, COUNT(user_id) as total_users, COUNT(DISTINCT(user_id))/(COUNT(user_id) + {}) as popularity_index FROM train_interactions GROUP BY recording_mbid ORDER BY average_plays DESC LIMIT 100'.format(beta))
        top_train_100 = [row['recording_mbid'] for row in popularity_train.select('recording_mbid').collect()]

        train_map_value = meanAveragePrecision(spark, train_df, top_train_100)
        val_map_value = meanAveragePrecision(spark, val_df, top_train_100)

        print(f'\nBeta: {beta}\nTrain mAP: {train_map_value}\nTest mAP: {val_map_value}')

        if val_map_value > best_test_map:
            best_beta = beta
            best_train_map, best_test_map = train_map_value, val_map_value
            top_recommendations = top_train_100

    print(f'\n\n\tBest Beta: {best_beta}\n\tBest Test mAP: {best_test_map}\n\tBest Train mAP: {best_train_map}')
    # print(f'Best Recommendations = {top_recommendations}\n')


def baseline_2(spark, train_df, val_df):

    train_df.createOrReplaceTempView('train_interactions')
    val_df.createOrReplaceTempView('val_interactions')

    best_beta = 0.0
    best_train_map, best_test_map = 0.0, 0.0
    top_recommendations, beta_values = [], [100, 1000, 5000, 10000, 50000, 100000]

    for beta in beta_values:

        popularity_train = spark.sql('SELECT recording_mbid, COUNT(user_id) as total_plays, ((COUNT(user_id)/(COUNT(DISTINCT(user_id)) + {})) * COUNT(DISTINCT(user_id))) as popularity_index FROM train_interactions GROUP BY recording_mbid ORDER BY average_plays DESC LIMIT 100'.format(beta))
        top_train_100 = [row['recording_mbid'] for row in popularity_train.select('recording_mbid').collect()]

        train_map_value = meanAveragePrecision(spark, train_df, top_train_100)
        val_map_value = meanAveragePrecision(spark, val_df, top_train_100)

        print(f'\nBeta: {beta}\nTrain mAP: {train_map_value}\nTest mAP: {val_map_value}')

        if val_map_value > best_test_map:
            best_beta = beta
            best_train_map, best_test_map = train_map_value, val_map_value
            top_recommendations = top_train_100

    print(f'\n\n\tBest Beta: {best_beta}\n\tBest Test mAP: {best_test_map}\n\tBest Train mAP: {best_train_map}')
    # print(f'Best Recommendations = {top_recommendations}\n')


def baseline_3(spark, train_df, val_df):

    train_df.createOrReplaceTempView('train_interactions')
    val_df.createOrReplaceTempView('val_interactions')

    best_beta = 0.0
    best_train_map, best_test_map = 0.0, 0.0
    top_recommendations, beta_values = [], [100, 1000, 5000, 10000, 50000, 100000]

    for beta in beta_values:

        popularity_train = spark.sql('SELECT recording_mbid, COUNT(user_id) as total_plays, (COUNT(DISTINCT(user_id)) + COUNT(user_id)/(COUNT(DISTINCT(user_id)) + {})) as popularity_index FROM train_interactions GROUP BY recording_mbid ORDER BY average_plays DESC LIMIT 100'.format(beta))
        top_train_100 = [row['recording_mbid'] for row in popularity_train.select('recording_mbid').collect()]

        train_map_value = meanAveragePrecision(spark, train_df, top_train_100)
        val_map_value = meanAveragePrecision(spark, val_df, top_train_100)

        print(f'\nBeta: {beta}\nTrain mAP: {train_map_value}\nTest mAP: {val_map_value}')

        if val_map_value > best_test_map:
            best_beta = beta
            best_train_map, best_test_map = train_map_value, val_map_value
            top_recommendations = top_train_100

    print(f'\n\n\tBest Beta: {best_beta}\n\tBest Test mAP: {best_test_map}\n\tBest Train mAP: {best_train_map}')
    # print(f'Best Recommendations = {top_recommendations}\n')


if __name__ == '__main__':

    spark = SparkSession.builder.appName('baseline_compare').getOrCreate()

    if len(sys.argv) == 2:
        data_type = sys.argv[1]
        spark, train_df, val_df = load_train_val(spark, data_type)

    if len(sys.argv) == 3:
        data_type, data_size = sys.argv[1], sys.argv[2]
        spark, train_df, val_df = load_train_val(spark, data_type, data_size)

    print('Baseline 1')
    baseline_1(spark, train_df, val_df)
    print('\n','---*---'*20,'\n')

    print('Baseline 2')
    baseline_2(spark, train_df, val_df)
    print('\n','---*---'*20,'\n')

    print('Baseline 3')
    baseline_3(spark, train_df, val_df)
    print('\n','---*---'*20,'\n')