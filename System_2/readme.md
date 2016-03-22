# 1 i-vector framework for NLP

i-vectors often seen in Speaker verification domain. It projects variable length speech utterances into a fixed-size low-
dimensional vector, namely i-vector. (More info on i-vector can be found in [Front-End Factor Analysis for Speaker Verification](https://www.researchgate.net/profile/Pierre_Dumouchel/publication/224166071_Front-End_Factor_Analysis_for_Speaker_Verification/links/0deec5176777115c24000000.pdf) )

This study propose use i-vector to model sentences for NLP.



This framework is developed using [Gensim](https://github.com/piskvorky/gensim) and  [Kaldi](https://github.com/kaldi-asr/kaldi)

Training of i-vector system is in a completely unsupervised manner, it includes training of word2vec and training of i-vector extractor. Evaluation is done on IMDB and SemEval 2016 Task4A.

# 2 Steps to build the system
## 2.1 Check [prerequistes](./prerequisites.md)

## 2.2 Train Gensim W2V model
Word-to-vector model is trained by using Gensim.

###2.2.1 Data used:
Train (25000) and unlabeled(50000) of imdb, you should have all the data if you clone this repository.
####Convert data to list of tokens

First convert imdb data to list of tokens for unsupervised learning of w2v.
```
python 0000_imdb_data_2_line_sentances.py
```
This will produce the merged labeled train+unlabled training data, which should contain 75000 movie reviews.

These two files will convert the test and all SemEval data to list of tokens.
```
0001_imdb_test_2_line_sentances.py
0002_semeval2016_tweet_2_line_sentance.py
```
The output files are saved in ./tempfolder

####Train word-2-vector

```
python 0010_w2v_train.py
```
This will train w2v based on data in previous session, with following setup:
```
    num_features = 20    # Word vector dimensionality
    min_word_count = 10 #25   # Minimum word count
    num_workers = 8       # Number of threads to run in parallel
    context = 7          # Context window size
    downsampling = 1e-3
```

Output model is ./tempfolder/imdb_trained_75000.w2v.20.bin

## 2.3 Convert text data to kaldi format
We already have all the data in "list of tokens" format, so we can use word-to-vector to convert them into list of vectors.
## 2.3.1 Convert text data to kaldi features
```
python 0020_w2v_to_kaldi.py
```

This process including converting training data, unlabeled and test data of imdb and train,dev,devtest and test set for SemEval2016.

The converted data will be stored in **data** folder and follow Kaldi's specification,[more on Kaldi data format](https://github.com/StevenLOL/Research_speech_speaker_verification_nist_sre2010/blob/master/doc/help_kaldi.md)..

In each folder there will be one w2vFeatures.ark and W2vFeatures.ark.mean

The w2vFeatures.ark is list of word vectors of sentences.

w2vFeatures.ark.mean is the mean w2v vector for each sentence.

Both can be opened via text editor.

### 2.3.2 Convert kaldi features to feats.ark,feats.scp and utt2spk

```
sh ./0030_refine_kaldi_datafile.sh
```

There will be three more files in each folder in ./data, namely feats.ark,feats.scp and utt2spk. These files are required by kaldi system.

1. feats.ark is feature's raw data
2. feats.scp is the description of feats.ark
3. utt2spk stores the utterance to speaker relationship for a Speaker Verification system. In this task it is useless, however it could used to store sentence to speaker/topic relationship as well, so we will keep this file in the system.


###Check point

After 2.3.2 , there will be six folders in ./data

./data/ivector_extractor_train contains training data for i-vector extractor training.

semeval_dev, semeval_devtest ,semeval_train contrains semeval data as their name imply.

##3 Training of i-vector extractor and i-vector extraction
Training and extraction are all done in following script:
This process will take hours to completed, however if you have a "super" computer or a cluster, you can use more threads and processes to train the system.

```
sh ./run.w2v.sh   

```
We are still polishing code here, if you want to retrain the i-vector system drop us an email: steven@aicyber.com
Meanwhile pre-trained i-vectors are given, so you can reproduce the scores we listed in the paper , see 4 Evaluation.

The output will be in a folder named **exp** .



##4 Evaluation
There are pre-trained i-vectors that you can play with, they are in file exp.zip under ./exp
Unzip the folders into ./exp

##4.1 Vector Space model baseline for IMDB

```
python system_2_baseline.py
```

And output will be:
```
             precision    recall  f1-score   support

          0     0.8928    0.8725    0.8825     12500
          1     0.8753    0.8952    0.8851     12500

avg / total     0.8840    0.8838    0.8838     25000
```

##4.2 i-vector for IMDB

```
python 0041_test_ivector_imdb.py
```
And output will be:
```
             precision    recall  f1-score   support

          0     0.8689    0.8838    0.8763     12500
          1     0.8817    0.8666    0.8741     12500

avg / total     0.8753    0.8752    0.8752     25000
```
##4.3 i-vector for SemEval2016

```
python 0042_test_ivector_SemEval2016.py
```
And output:
```
'../semeval2016/100_topics_100_tweets.sentence-three-point.subtask-A.dev.gold.txt.full

37.32

../semeval2016/100_topics_100_tweets.sentence-three-point.subtask-A.devtest.gold.txt.full
38.14

```
#FAQ
## Is your imdb data the same as those published work?
Yes, it is same, same training, testing, and unsupervised data.
## What is the delta ?
One can say ' word vector - next word vector ' in general , details are in [DeltaFeatures] (http://kaldi.sourceforge.net/classkaldi_1_1DeltaFeatures.html)
## What is the major task befor i-vector training?
Convert those data into kaldi format...
## How long the i-vector training could completed?
On my desktop with 16GB memory it will take 6 hours for training to completed . 
If you have a cluster , you can make training done with less time. 

## How could I edit python file?

Any text editor or [PyCharm](https://www.jetbrains.com/pycharm/)
