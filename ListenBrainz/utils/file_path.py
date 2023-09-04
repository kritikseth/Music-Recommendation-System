from .constants import (
    HDFS_DATA, GREENE_DATA, HDFS_USER_DATA, GREENE_USER_DATA
)

get_hdfs_path = lambda file: f'{HDFS_DATA}/{file}.parquet'
get_greene_path = lambda file: f'{GREENE_DATA}/{file}.parquet'
get_hdfs_user_path = lambda file: f'{HDFS_USER_DATA}/{file}.parquet'
get_greene_user_path = lambda file: f'{GREENE_USER_DATA}/{file}.parquet'