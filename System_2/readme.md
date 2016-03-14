#i-vector framework for NLP

i-vector system is trained on training+unlabeled imdb data, it is then evaluated on the IMDB as well as SemEval2016 tweets.

#Steps to build the system

##Check [prerequistes](./prerequisites.md)

##Train Gensim W2V model
###Data used:
train unlabeled of imdb
Training of a word to vector model by using Gensim.
####0000

First convert imdb data to list of tokens for unsupervised learning of w2v.
~~~
python 0000_imdb_data_2_line_sentances.py
~~~
This will produce the merged labeled train+unlabled training data, which should contain 75000 movie reviews.

These two files will convert , the test and all SemEval data to list of tokens.
~~~
0001_imdb_test_2_line_sentances.py
0002_semeval2016_tweet_2_line_sentance.py
~~~

####0010

~~~
python 0010_w2v_train.py
~~~
This will train w2v based on data in 0000 , with following setup:
~~~
    num_features = 20    # Word vector dimensionality
    min_word_count = 10 #25   # Minimum word count
    num_workers = 8       # Number of threads to run in parallel
    context = 7          # Context window size
    downsampling = 1e-3
~~~

Output model is ./tempfolder/trainAndUnalbed.w2v.20.bin

##Convert text data to kaldi format
We already have all the data in list of tokens format, so we can use w2v to convert list of tokens into list of vectors.

~~~
python 0020_w2v_to_kaldi.py
~~~

This process including converting training data, unlabeled and test data of imdb and train,dev,devtest and test set for SemEval2016.


##


The covnerted data will be stored in data folder and follow Kaldi's specification,[More on Kaldi data format](https://github.com/StevenLOL/Research_speech_speaker_verification_nist_sre2010/blob/master/doc/help_kaldi.md)..

This process will take time to completed.

In each folder there will be on w2vFeatures.ark and W2vFeatures.ark.mean

The w2vFeatures.ark is list of word vectors of sentences.

w2vFeatures.ark.mean is the mean w2v vector for each sentence.

Both can be opened via text editor.




##Training of i-vector extractor
###Data used:
train unlabeled of imdb
##Extract i-vectors
###Data used:
all data which suit the supervised training


