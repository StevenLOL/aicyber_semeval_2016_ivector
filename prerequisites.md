#Hardware

Computer with > 12 GB memory is required for training of i-vector extractor

#Software

The use of a linux system is assumed.

##Gensim
For training word to vector
###Via pip
~~~
pip install gensim
~~~
##Kaldi

For training i-vector extractor

###Clone kaldi
https://github.com/kaldi-asr/kaldi
###Build kaldi follow instructions in INSTALL
~~~
(1)
go to tools/  and follow INSTALL instructions there.

(2)
go to src/ and follow INSTALL instructions there.
~~~
###Setup KALDI_ROOT to point to your kaldi folder in path.sh
Eg: export KALDI_ROOT=/home/steven/gits/kaldi
###Verify kaldi
In this folder
~~~
. path
copy-feats
~~~
Something will print out rather than "copy-feats: command not found"

##scikit-learn
~~~
pip install scikit-learn
~~~




