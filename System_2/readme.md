# 1 i-vector framework for NLP

i-vectors often seen in Speaker verification domain. It projects variable length speech utterances into a fixed-size low-
dimensional vector, namely i-vector. (More info on i-vector can be found in [Front-End Factor Analysis for Speaker Verification](https://www.researchgate.net/profile/Pierre_Dumouchel/publication/224166071_Front-End_Factor_Analysis_for_Speaker_Verification/links/0deec5176777115c24000000.pdf) )

This study propose use i-vector to model sentances. 

Training of i-vector system is in a completely unsupervised manner, it includes training of word2vec and training of i-vector extractor. Evaluation is done on IMDB similar and SemEval 2016 Task4A.


# 2 Steps to build the system
## 2.1 Check [prerequistes](./prerequisites.md)

## 2.2 Train Gensim W2V model
###2.2.1 Data used:
train unlabeled of imdb
Training of a word to vector model by using Gensim.
####0000

First convert imdb data to list of tokens for unsupervised learning of w2v.
~~~
python 0000_imdb_data_2_line_sentances.py
~~~
This will produce the merged labeled train+unlabled training data, which should contain 75000 movie reviews.

These two files will convert the test and all SemEval data to list of tokens.
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

## 2.3 Convert text data to kaldi format
We already have all the data in list of tokens format, so we can use w2v to convert them into list of vectors.

~~~
python 0020_w2v_to_kaldi.py
~~~

This process including converting training data, unlabeled and test data of imdb and train,dev,devtest and test set for SemEval2016.


##


The covnerted data will be stored in **data** folder and follow Kaldi's specification,[More on Kaldi data format](https://github.com/StevenLOL/Research_speech_speaker_verification_nist_sre2010/blob/master/doc/help_kaldi.md)..

This process will take time to completed.

In each folder there will be one w2vFeatures.ark and W2vFeatures.ark.mean

The w2vFeatures.ark is list of word vectors of sentences.

w2vFeatures.ark.mean is the mean w2v vector for each sentence.

Both can be opened via text editor.




##Training of i-vector extractor
###Data used:
train unlabeled of imdb
##Extract i-vectors
###Data used:
all data which suit the supervised training


#FAQ
## Is your imdb data the same as those published work?
Yes, it is same, same training, testing, and unsupvised data.
## What is the delta or double deta ?
One can say it's this word vector - next word vector in general , details is in [DeltaFeatures] (http://kaldi.sourceforge.net/classkaldi_1_1DeltaFeatures.html)
## What is the major task, befor i-vector training?
Convert those data into kaldi format...
## How long the i-vector training could completed?
On my desktop with 16GB memory it will take 6 hours to completed training.

