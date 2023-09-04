import getpass

NETID = getpass.getuser()
HDFS_DATA = '/user/bm106_nyu_edu/1004-project-2023'
GREENE_DATA = '/scratch/work/courses/DSGA1004-2021/ListenBrainz'
HDFS_USER_DATA = f'hdfs:/user/{NETID}'
GREENE_USER_DATA = f'/home/{NETID}'