# -*- coding: utf-8 -*-

# Import packages and config
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import datetime
import csv
import os
# Takes tweets and a designated csv file and writes them to it.

class StdOutListener(StreamListener):
    def __init__(self, max_count):
        self.tweet_count = 1
        self.max_count = max_count
        super(StdOutListener, self).__init__()

    def on_status(self, status):
        
        if self.tweet_count > self.max_count:
            return False
        else:
            if (status.lang == "en") and status.user.followers_count >= 100:
                if(hasattr(status, 'retweeted_status')
                  or hasattr(status, 'quoted_status_id')
                  or status.in_reply_to_status_id
                  or status.in_reply_to_user_id
                  or status.in_reply_to_screen_name):
                      pass
                else:
                    # Creating this formatting so when exported to csv the tweet stays on one line
                    tweet_text = "'" + status.text.replace('\n', ' ') + "'"
                    csvwriter_retweet_count.writerow([status.id,status.retweet_count])
                    csvwriter_user_info.writerow([status.id, tweet_text, status.user.id, status.user.friends_count,
                                         status.user.followers_count, status.user.created_at.strftime('%Y-%m-%d'),
                                         status.user.statuses_count, status.user.favourites_count])

                    if(self.tweet_count%100 == 0):
                        print(status.id, self.tweet_count)
                    self.tweet_count += 1
                    return True

    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_error disconnects the stream
            return False

if __name__ == '__main__':

    max_count = 1500           # number of tweets

    listener = StdOutListener(max_count)
    auth = OAuthHandler("CONSUMER_KEY", "CONSUMER_SECRET")
    auth.set_access_token("ACCESS_TOKEN", "ACCESS_TOKEN_SECRET")
    stream = Stream(auth, listener)
#    path = os.getcwd()
    path = os.path.dirname(os.path.realpath(__file__))
    keywords = []
    f = open(path+"/keywords.txt", "r")
    for line in f:
        keywords.append(line.split('  ')[0])
    # print(keywords)
    f.close()
    # Filter based on listed items
    user_info = open(path+'/user_info_t1m1.csv','w')
    temporal_retweet_count = open(path+'/temporal_retweet_count_t1m1.csv','w')

    csvwriter_user_info = csv.writer(user_info)
    csvwriter_retweet_count = csv.writer(temporal_retweet_count)

    user_info_fields = ['tweet_id','text','user_id','friends_count','followers_count',
                        'account_age','total_tweet_count','favourited_tweet_count']
    retweet_count_fields = ['tweet_id',0]
    
    csvwriter_retweet_count.writerow(retweet_count_fields)
    csvwriter_user_info.writerow(user_info_fields)

    stream.filter(track=keywords)    # This is topic specific
    user_info.close()
    temporal_retweet_count.close()
