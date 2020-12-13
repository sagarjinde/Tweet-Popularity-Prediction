# Temporal Retweet Prediction

## Objective

- Build a model that uses textual data as well as the author's social features to predict the potential influence of tweets, as
measured by their retweet counts
- Predict retweet count each hour for a duration of 72 hours

**Note:** It's a regression based problem.

## Get Started

The model is a modified version of [Retweet Wars](https://www.cs.unc.edu/~mbansal/papers/retweetwars-wacv18.pdf)

Go through the [report](https://github.com/sagarjinde/Tweet-Popularity-Prediction/blob/master/report.pdf) for detailed explanation of the model

#### Model while training:

![Training Model](https://github.com/sagarjinde/Tweet-Popularity-Prediction/blob/master/figs/train_fig.png)

#### Model while testing:

![Testing Model](https://github.com/sagarjinde/Tweet-Popularity-Prediction/blob/master/figs/test_fig.png)

Methodology Explained (video) :- https://www.youtube.com/watch?v=albemFlWzBI

## Running the code

`Python version: 3.6.8`

### Create a virtual environment (Recommended but optional)
Install virtualenv  : `sudo apt-get install virtualenv` </br>
Create virtualenv   : `virtualenv --system-site-packages -p python3 ./venv` </br>
Activate virtualenv : `source venv/bin/activate` </br>

### Install Requirements
Run `pip3 install -r requirements.txt`

**Note:** If you get `ModuleNotFoundError: No module named 'matplotlib._path'` error, run `pip3 install --upgrade matplotlib`

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
   **Note:** We have created corona specific glove embedding called `custom_WE.txt`, so if you just want to test run the code, you DON'T have to 
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

### Make alteast 2 Twitter developer accounts
Developer accounts are required in order to collect data.

### Collect user info whose tweet match any one of the keywords
Run `extract_tweets.py`. Once run, it will collect 1500 tweets according to keywords. I added an additional condition that the user should 
have atleast 100 followers so that there is a greater change of not getting all 0's in `temporal_retweet_count`.

### Get number of retweets each hour
Run `get_retweet_count.py` to get the retweet count of above 1500 tweets. </br>
**Note:** This will give only the retweet count at the time you run this code. </br>

If you want it to run each hour automatically, use cron. </br>
**Note:** If the tweet is deleted, then NULL is added to csv. </br>