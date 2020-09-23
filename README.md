Methodology Explained (video) :- https://www.youtube.com/watch?v=albemFlWzBI

#################### Temporal Retweet Prediction ####################

Python version: 3.6.8

===> CREATE A VIRTUAL ENVIRONMENT (Recommended but optional)
Install virtualenv  : sudo apt-get install virtualenv
Create virtualenv   : virtualenv --system-site-packages -p python3 ./venv
Activate virtualenv : source venv/bin/activate

===> INSTALL REQUIREMENTS
run 'pip3 install -r requirements.txt'

NOTE: If you get this error 'ModuleNotFoundError: No module named 'matplotlib._path', run 'pip3 install --upgrade matplotlib'

===> HOW TO PREDICT RETWEET COUNT FOR YOUR TWEET
Run 'python predict_my_retweet.py' and enter tweet, friends_count, followers_count, account_age, total_tweet_count, favourited_tweet_count 
when asked by the program.

===> HOW TO TRAIN MODEL
1) Create word embedding vectors of 100-dimension for your tweet dataset by using glove embeddings code specified here
  'https://github.com/stanfordnlp/GloVe' and name it 'custom_WE.txt'. 
   NOTE: We have created 'corona' specific glove embedding called 'custom_WE.txt', so if you want to just run the code, you DONT have to 
   create new coutom word embedding.
2) Run 'python warm_up_lstm.py' to warm up the LSTM. This will create encoder model which will be saved in '/saved_models' and will be named 
   'encoder_model.h5'
3) Run 'python warm_up_drnn.py' to warm up the dynamic rnn. This will create decoder model which will be saved in '/saved_models' and will be 
   named 'decoder_model.h5'
4) Run 'python end-to-end.py' to train the full model from end to end. This will create full model which will be saved in '/saved_models' and 
   will be named 'final_model.h5'

#################### Data Creation ####################

===> Create keywords
Create a file named 'keywords.txt' and enter keywords related to tweets that you want to extract.

===> Make alteast 2 twitter developer accounts
Developer accounts are required in order to collect data.

===> Collect user info who tweet match any one of the keywords
run 'extract_tweets.py'. once run, it will collect 1500 tweets according to keywords. We added an additional condition that the user should 
have atleast 100 followers so that there is a greater change of not getting all 0's in temporal_retweet_count.

===> Get number of retweets each hour
run 'get_retweet_count.py' to get the retweet count of above 1500 tweets. 
NOTE: This will give only the retweet count at the time you run this code.

If you want it to run each hour, use cron.
NOTE: If the tweet is deleted, then NULL is added to csv.

#################### Create GloVe Embedding ####################




