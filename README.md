Methodology Explained (video) :- https://www.youtube.com/watch?v=albemFlWzBI

# Temporal Retweet Prediction

## Running the code

`Python version: 3.6.8`

### Create a virtual environment (Recommended but optional)
Install virtualenv  : `sudo apt-get install virtualenv` </br>
Create virtualenv   : `virtualenv --system-site-packages -p python3 ./venv` </br>
Activate virtualenv : `source venv/bin/activate` </br>

### Install Requirements
run `pip3 install -r requirements.txt`

NOTE: If you get this error `ModuleNotFoundError: No module named 'matplotlib._path'`, run `pip3 install --upgrade matplotlib`

### How to predict retween count for your tweet
Run `python predict_my_retweet.py` and enter 
- tweet
- friends_count
- followers_count
- account_age
- total_tweet_count
- favourited_tweet_count 

### How to train model 
- Download pre-trained twitter word embedding from https://nlp.stanford.edu/projects/glove/
- Create word embedding vectors of 100-dimension for your tweet dataset by using code specified in https://github.com/stanfordnlp/GloVe and name it `custom_WE.txt`. </br>
   NOTE: We have created corona specific glove embedding called `custom_WE.txt`, so if you just want to test run the code, you DONT have to 
   create new coutom word embedding.
- Run `python warm_up_lstm.py` to warm up the LSTM. This will create encoder model which will be saved in `/saved_models` and will be named 
   `encoder_model.h5`
- Run `python warm_up_drnn.py` to warm up the dynamic RNN. This will create decoder model which will be saved in `/saved_models` and will be 
   named `decoder_model.h5`
- Run `python end-to-end.py` to train the full model from end to end. This will create full model which will be saved in `/saved_models` and 
   will be named `final_model.h5`

## Custom dataset

### Create keywords
Create a file named `keywords.txt` and enter keywords related to tweets that you want to extract.

### Make alteast 2 twitter developer accounts
Developer accounts are required in order to collect data.

### Collect user info whose tweet match any one of the keywords
run `extract_tweets.py`. Once run, it will collect 1500 tweets according to keywords. I added an additional condition that the user should 
have atleast 100 followers so that there is a greater change of not getting all 0's in `temporal_retweet_count`.

### Get number of retweets each hour
run `get_retweet_count.py` to get the retweet count of above 1500 tweets. </br>
NOTE: This will give only the retweet count at the time you run this code. </br>

If you want it to run each hour, use cron. </br>
NOTE: If the tweet is deleted, then NULL is added to csv. </br>