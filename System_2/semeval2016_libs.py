import codecs
from tweetokenize import Tokenizer
import warnings
import os
import configure
def load_semeval_text_only(fname,delimiter=u'\t'):
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
    return ids,y,x



#scoring

from subprocess import Popen, PIPE, STDOUT
def scoreit(ids,predict,goldenfile):
    tempfile='./.temp.txt'
    with codecs.open(tempfile,'w',encoding='utf-8') as fout:
        for id,value in zip(ids,predict):
            fout.write(u'%s\t%s\n'%(id,value))
    slave = Popen(['perl', configure.SCORE_SCRIPT,tempfile,goldenfile], stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output=slave.stdout.readlines()
    os.remove(tempfile)
    return output

def scoreSameOrder(predict,goldenfile):
    warnings.warn('predict value must have same order as %s'%(goldenfile))
    ids,dumy1,dumy2=load_semeval_text_only(goldenfile)
    tempfile='./.temp.txt'
    with codecs.open(tempfile,'w',encoding='utf-8') as fout:
        for id,value in zip(ids,predict):
            fout.write(u'%s\t%s\n'%(id,value))
    slave = Popen(['perl', configure.SCORE_SCRIPT,tempfile,goldenfile], stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output=slave.stdout.readlines()
    os.remove(tempfile)
    return output

