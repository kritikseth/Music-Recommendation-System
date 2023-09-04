import sys
from .constants import NETID
from .file_path import (
    get_greene_user_path, get_hdfs_user_path
)
import pandas as pd
from pyspark.sql import SparkSession

def load_train_val(spark=None):
    
    file_type, file_size = 'ordered', ''

    if '--ordered' in sys.argv:
        file_type = 'ordered'
        if '--small' in sys.argv:
            file_size = '_small'

    elif '--random' in sys.argv:
        file_type = 'random'
        if '--small' in sys.argv:
            file_size = '_small'
    
    train_file_name = f'interactions_train_{file_type}{file_size}'
    val_file_name = f'interactions_val_{file_type}{file_size}'

    if '--n' in sys.argv:
        train_file_name = f'train_{file_type}{file_size}'
        val_file_name = f'val_{file_type}{file_size}'

    if NETID.endswith('_nyu_edu'):
        train_df = spark.read.parquet(get_hdfs_user_path(train_file_name))
        val_df = spark.read.parquet(get_hdfs_user_path(val_file_name))
        
        print('Printing train inferred schema:\n')
        train_df.printSchema()
    else:
        train_df = pd.read_parquet(get_greene_user_path(train_file_name))
        val_df = pd.read_parquet(get_greene_user_path(val_file_name))

    return train_df, val_df

def load_train_test(spark=None):
    
    train_file_name = f'train'
    test_file_name = f'test'

    if NETID.endswith('_nyu_edu'):
        train_df = spark.read.parquet(get_hdfs_user_path(train_file_name))
        test_df = spark.read.parquet(get_hdfs_user_path(test_file_name))
        
        print('Printing train inferred schema:\n')
        train_df.printSchema()
    else:
        train_df = pd.read_parquet(get_greene_user_path(train_file_name))
        test_df = pd.read_parquet(get_greene_user_path(test_file_name))

    return train_df, test_df