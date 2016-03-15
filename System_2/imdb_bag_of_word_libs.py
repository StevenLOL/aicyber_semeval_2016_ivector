#coding=utf-8
import configure
import pandas as pd
import os
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import codecs
import csv
from tweetokenize import Tokenizer
gettokens=Tokenizer()
def loadData(fin):
    data=csv.reader(file(fin,'r'))
    return data


def loadUnlabedData():
    # Read data from files
    train = pd.read_csv(os.path.join(configure.IMDB_BOW_DATA_FOLD, "labeledTrainData.tsv"), header=0,
                        delimiter="\t", quoting=3, encoding='utf-8')
    test = pd.read_csv(os.path.join(configure.IMDB_BOW_DATA_FOLD, "testData.tsv"), header=0, delimiter="\t", quoting=3, encoding='utf-8')
    unlabeled_train = pd.read_csv(os.path.join(configure.IMDB_BOW_DATA_FOLD, "unlabeledTrainData.tsv"), header=0,
                                  delimiter="\t", encoding='utf-8', quoting=3)

    # Verify the number of reviews that were read (100,000 in total)
    print "Read %d labeled train reviews, %d labeled test reviews, " \
    "and %d unlabeled reviews\n" % (train["review"].size,
    test["review"].size, unlabeled_train["review"].size )
    return train,test,unlabeled_train


def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    words=[w for w in words if w!=None]
    return(words)
discardChar='\xc2\x85'.decode('utf-8')
def processLine(lineinput):
    rv=gettokens.tokenize(lineinput)
    #rv.remove('\xc2\x85'.decode('utf-8'))
    #rv=[s.lower() for s in rv]
    rv=[s.lower() for s in rv if s!=discardChar]
    return rv

def processLine_old(lineinput):
    #print lineinput

    linetext= BeautifulSoup(lineinput).get_text()
    linetext=filterlineEmoji(linetext)
    linetext=filterline(linetext)
    return linetext

replaceEmoji={u"you'd":u"you had", u"you'll":u"you will", u"you're":u"you are",u'cannot':u' cant ',u"can't":u" cant ",u';)':u' good ',u':)':u' good ',u':P':u' good ',u':D':u' good ',u'^_^':u' good ',u'*_*':u' good ',u'*_*':u' good ',u':>':u' good ',u':-)':u' good ',u':-D':u' good ',u':-p':u' good ',u'(:':u' good ',u'(;':u' good ',u':->':u' good ',u':-O':u' good ',u':-o':u' good ',u':<':u' sad ',u'>_<':u' sad ',u':(':u' sad ',u':-(':u' sad ',u':/ ':u' sad '}
replaceDict={u'\\u002c':u' ',u'\\u2019':u'\'',u'\\"':u'',u'\\""':u'',u'...':u' ...',u'. ':u' . ',u'.\n':u' .',u'."':u' .',u'!':u' ! ',
             u'?':u' ? ',
             u': ':u' ',u'\' ':u' ',u' \'':u' ',u'"':u'',u',':u' ',u'(':u'',u')':u'',u'#':u'',u'@':u''}

def filterlineEmoji(linein):
    assert type(linein)==unicode
    lineout=linein
    for c in replaceEmoji:
        #print c
        if c in lineout:
            lineout=lineout.replace(c,replaceEmoji[c])
    #lineout=re.sub(reg,u" ", lineout)   #lineout.replace(reg,u' ')
    #lineout= re.sub(u"[^a-zA-Z]",u" ", lineout)
    while u'  ' in lineout:
        lineout=lineout.replace(u'  ',u' ')

    return lineout

    #return lineout.replace(u'_',u' ')
def filterline(linein):
    assert type(linein)==unicode
    lineout=linein
    for c in replaceDict:
        #print c
        if c in lineout:
            lineout=lineout.replace(c,replaceDict[c])
    #lineout=re.sub(reg,u" ", lineout)   #lineout.replace(reg,u' ')
    #lineout= re.sub(u"[^a-zA-Z]",u" ", lineout)
    lineout=lineout.replace(u'\\',u' ')
    lineout=lineout.replace(u'/',u' ')
    lineout = re.sub(u"[^a-zA-Z ]",u" ", lineout)
    while u'  ' in lineout:
        lineout=lineout.replace(u'  ',u' ')
    #words=lineout.lower().split(u' ')
    #processdwords=map(wordnetLemmatizer.stem,words)
    #lineout=u' '.join(processdwords)

    return lineout.lower()



def kaldiID_2_LB(idlist):
    '''
    convert well formated kaldi ids into  lists of id  and label
    :param idlist: a list of kaldi id
    :return: a list of labels
    '''
    rv=[]
    for id in idlist:

        v=id.split('_')[1]
        rv.append(v)
        '''
        if v in ['1','2','3']:
            rv.append(0)
        elif v=='5':
            rv.append(1)
        else:
            print id, 'Error'
        '''
    return rv


def loadFeatsText(fname):
    '''
    load kaldi feats
    :param fname: file name of a kaldi feat
    :return:
    two lists
    ids, and values
    '''
    rvvalues=[]
    fdata=open(fname).read()
    fdata=fdata.replace('[\n','[').replace('[ \n','[')
    fdata=fdata.split('\n')
    fdata=[s.replace('[','').replace(']','').replace('  ',' ').replace('  ',' ').strip() for s in fdata if len(s)>10 and '_22 ' not in s ]
    print fdata[0]
    rvids=[s.split()[0] for s in fdata]
    values=[map(float,s.split()[1:]) for s in fdata]
    return rvids,values

