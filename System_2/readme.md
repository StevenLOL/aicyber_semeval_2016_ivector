#i-vector framework for NLP

i-vector system is trained on training+unlabeled imdb data, it is then evaluated on the IMDB as well as SemEval2016 tweets.

#Steps to build the system

##Check [prerequistes](./prerequisites.md)

##Train Gensim W2V model
###Data used:
train unlabeled of imdb
Training of a word to vector model by using Gensim.


##Convert text data to kaldi format

Including train unlabeled test of imdb and train dev devtest and test for SemEval2016

The covnerted data will be stored in data folder

[More on Kaldi data format](https://github.com/StevenLOL/Research_speech_speaker_verification_nist_sre2010/blob/master/doc/help_kaldi.md).

##Training of i-vector extractor
###Data used:
train unlabeled of imdb
##Extract i-vectors
###Data used:
all data wich suit the supersed training


