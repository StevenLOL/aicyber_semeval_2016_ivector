#coding=utf8
import os
from bs4 import BeautifulSoup
import codecs
import path
import re
from nltk.stem.lancaster import LancasterStemmer
#encoding=utf-8
import configure

from tweetokenize import Tokenizer

wordnetLemmatizer=LancasterStemmer()
from nltk.sentiment.util import mark_negation
from nltk.sentiment import vader
stAnalyzer=vader.SentimentIntensityAnalyzer()
from nltk.corpus import stopwords
replaceEmoji={u"you'd":u"you had", u"you'll":u"you will", u"you're":u"you are",u'cannot':u' cant ',u"can't":u" cant ",u';)':u' good ',u':)':u' good ',u':P':u' good ',u':D':u' good ',u'^_^':u' good ',u'*_*':u' good ',u'*_*':u' good ',u':>':u' good ',u':-)':u' good ',u':-D':u' good ',u':-p':u' good ',u'(:':u' good ',u'(;':u' good ',u':->':u' good ',u':-O':u' good ',u':-o':u' good ',u':<':u' sad ',u'>_<':u' sad ',u':(':u' sad ',u':-(':u' sad ',u':/ ':u' sad '}
replaceDict={u'\\u002c':u' ',u'\\u2019':u'\'',u'\\"':u'',u'\\""':u'',u'...':u' ...',u'. ':u' . ',u'!':u' ! ',
             u'?':u' ? ',
             u': ':u' ',u'\' ':u'',u' \'':u'',u'"':u'',u',':u' ',u'(':u'',u')':u'',u'#':u'',u'@':u''}
#
print replaceDict
#u'#':u''
reg=u'[a-zA-z]+://[^\s]*'
#reg='\u[^\s]*'

#print re.sub(reg,' ',uc)

#def appFeats(tfidfFeatures,)

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
def filterNegation(linein):
    assert type(linein)==unicode
    lineout=linein
    parts=linein.split(u' ')
    markednageation=mark_negation(parts)
    lineout= u' '.join(markednageation)
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
    while u'  ' in lineout:
        lineout=lineout.replace(u'  ',u' ')
    #words=lineout.lower().split(u' ')
    #processdwords=map(wordnetLemmatizer.stem,words)
    #lineout=u' '.join(processdwords)
    return lineout.lower()


def load_semeval(fname,delimiter=u'\t'):
    assert os.path.isfile(fname)
    rvdata=codecs.open(fname,encoding='utf-8').readlines()
    assert type(rvdata[0])==unicode
    rvdata=[s.strip() for s in rvdata]
    assert len(rvdata)>0
    numberofColoums=len(rvdata[0].split(delimiter))
    ids=[s.split(delimiter)[0] for s in rvdata]
    if numberofColoums==3:
        y=[s.split(delimiter)[1] for s in rvdata]
        x=[s.split(delimiter)[2] for s in rvdata]

    elif numberofColoums==4:
        y=[s.split(delimiter)[2] for s in rvdata]
        x=[s.split(delimiter)[3] for s in rvdata]
    else:
        xstartindex=[int(s.split(delimiter)[2]) for s in rvdata]
        xendindex=[int(s.split(delimiter)[3]) for s in rvdata]
        y=[s.split(delimiter)[4] for s in rvdata]
        x=[s.split(delimiter)[5] for s in rvdata]
        print rvdata[0]
        x=[u' '.join(s.split(u' ')[xs:xe+1]) for s,xs,xe in zip(x,xstartindex,xendindex)]

    x=map(filterlineEmoji,x)
    #x=map(filterNegation,x)
    x=map(filterline,x)
    print 'totoal',len(rvdata) ,'@',fname
    print ids[0],y[0],x[0]
    #print x[0]
    return ids,y,x
def load_semeval2(fname):
    #what?!
    assert os.path.isfile(fname)
    rvdata=open(fname).readlines()
    rvdata=[s.strip() for s in rvdata]
    rvdata=open(fname).readlines()
    rvdata=[s.strip() for s in rvdata]
    for l in rvdata:
        parts=l.split('\t')
        yield parts[0],parts[1],parts[2]

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
    return(words)



def load_semeval_text_only(fname,delimiter=u'\t'):
    '''
    :param fname: file name
    :param delimiter: deliminater
    :return:
    id: all ids
    y:  training labels
    x:  training text , lower cased , and filtered text
    '''
    gettokens = Tokenizer()
    assert os.path.isfile(fname)
    rvdata=codecs.open(fname,encoding='utf-8').readlines()
    assert type(rvdata[0])==unicode
    rvdata=[s.strip() for s in rvdata]
    assert len(rvdata)>0
    numberofColoums=len(rvdata[0].split(delimiter))
    ids=[s.split(delimiter)[0] for s in rvdata]
    if numberofColoums==3:
        y=[s.split(delimiter)[1] for s in rvdata]
        x=[s.split(delimiter)[2] for s in rvdata]

    elif numberofColoums==4:
        y=[s.split(delimiter)[2] for s in rvdata]
        x=[s.split(delimiter)[3] for s in rvdata]
    else:
        xstartindex=[int(s.split(delimiter)[2]) for s in rvdata]
        xendindex=[int(s.split(delimiter)[3]) for s in rvdata]
        y=[s.split(delimiter)[4] for s in rvdata]
        x=[s.split(delimiter)[5] for s in rvdata]
        print rvdata[0]
        x=[u' '.join(s.split(u' ')[xs:xe+1]) for s,xs,xe in zip(x,xstartindex,xendindex)]

    print 'totoal',len(rvdata) ,'@',fname
    print ids[0],y[0],x[0]
    x=map(filterlineEmoji,x)
    x=map(unicode.lower,x)
    return ids,y,x



#scoring

from subprocess import Popen, PIPE, STDOUT
def scoreit(ids,predict,goldenfile,score_script=configure.SCORE_SCRIPT_DEV):
    '''

    :param ids:                 input ids
    :param predict:             input predict values
    :param goldenfile:          ref   values
    :param score_script:        perl scoring script from SemEval 2016
    :return:
    some print out, and *.scored files produced by score_script file.
    '''
    tempfile='./.temp.txt'
    with codecs.open(tempfile,'w',encoding='utf-8') as fout:
        for id,value in zip(ids,predict):
            fout.write(u'%s\t%s\n'%(id,value))
    print 'perl', score_script,tempfile,goldenfile
    slave = Popen(['perl', score_script,tempfile,goldenfile], stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output=slave.stdout.readlines()
    os.remove(tempfile)
    return output



