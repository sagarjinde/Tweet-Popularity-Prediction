# -*- coding: utf-8 -*-

import pandas as pd
import tweepy
import os
from tweepy import OAuthHandler


auth = OAuthHandler("CONSUMER_KEY", "CONSUMER_SECRET")
auth.set_access_token("ACCESS_TOKEN", "ACCESS_TOKEN_SECRET")
api = tweepy.API(auth, wait_on_rate_limit=True)

auth1 = OAuthHandler("CONSUMER_KEY1", "CONSUMER_SECRET1")
auth1.set_access_token("ACCESS_TOKEN1", "ACCESS_TOKEN_SECRET1")
api1 = tweepy.API(auth1, wait_on_rate_limit=True)

path = os.path.dirname(os.path.realpath(__file__))
retweet_file = open(path+'/temporal_retweet_count_t1m1.csv','r')
header = retweet_file.readline()   # skip the first line
present_index = len(header.split(','))-1
retweet_count_list = []

# for row in retweet_file:
for i in range(750):
    row = retweet_file.readline()
    status_id = int(row.split(',')[0])
    try:
        status = api.get_status(status_id)
        retweet_count_list.append(status.retweet_count)
    except tweepy.TweepError:
        retweet_count_list.append('NULL')

for j in range(750):
    row = retweet_file.readline()
    status_id = int(row.split(',')[0])
    try:
        status = api1.get_status(status_id)
        retweet_count_list.append(status.retweet_count)
    except tweepy.TweepError:
        retweet_count_list.append('NULL')
  
df = pd.read_csv(path+"/temporal_retweet_count_t1m1.csv")
df[present_index] = retweet_count_list
df.to_csv(path+'/temporal_retweet_count_t1m1.csv', index=False)
retweet_file.close()
