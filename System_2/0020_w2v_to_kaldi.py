#coding=utf-8
import numpy as np
import os
#import threading
import multiprocessing
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec,Word2Vec
from gensim import corpora, models
import logging
import codecs
import configure
import time
#load w2v model



#load text data


#rewrite to ark file

#create ark and scp file

#get features for ivector extractor training

def rewrite_w2v_2_kaldi(inputFileName,outputFileName,w2vmodel,lineID_Prefix=u'',haslabel=False):
    oovdict=dict()
    fin=codecs.open(inputFileName,'r','utf-8').read()
    fin=fin.replace('\xc2\x85'.decode('utf-8'),u'')
    fin=fin.split(u'\n')
    fin=[s for s in fin if len(s)>0]
    fout=codecs.open(outputFileName,'w','utf-8')
    meanOut=codecs.open(outputFileName+'.mean','w','utf-8')
    linecount=0
    t1=time.time()
    #while True:
    print '#of line= ',len(fin),inputFileName
    for line in fin:
        #line=fin.readline()
        #if not line:break
        lineFeatures=u''
        lb=u''
        meanVecter=None
        meanCount=0
        for wordindex,word in enumerate( line.split(u' ')):
            w=word.strip()
            if haslabel and wordindex==0:
                lb=w
                if u'星' in w:
                    lb=w.replace(u'星',u'')
                continue
            if len(w)>0:

                #if not w in w2vmodel:
                #    w=w.lower()
                if w in w2vmodel:
                    vectors=np.array(w2vmodel[w])
                    #print type(vectors),vectors.shape,np.array(vectors)
                    #meanVecter=np.array(vectors)
                    #print meanVecter.shape
                    if meanCount==0:
                        meanVecter=vectors
                        #print 'set zero'
                    else:
                        meanVecter+=vectors
                    meanCount+=1
                    lineFeatures+=u'\n'
                    for v in vectors:
                        lineFeatures+=u'%f '%(v)
                else:
                    if w not in oovdict:
                        #print 'OOV',w
                        oovdict[w]=1
                    else:
                        oovdict[w]+=1


        #words=[u'%f'%(s) for sindex,s in ]
        if meanCount>0:
            lout=u'%s%020d_%s [ %s ]\n'%(lineID_Prefix,linecount,lb,lineFeatures)
            fout.write(lout)

            lineFeatures=u'\n'
            meanVecter=meanVecter/meanCount
            for v in meanVecter:
                lineFeatures+=u'%f '%(v)

            lout=u'%s%020d_%s [ %s ]\n'%(lineID_Prefix,linecount,lb,lineFeatures)
            meanOut.write(lout)

            linecount+=1
            if linecount % 1000==0:
                print linecount,time.time()-t1

        #if linecount>1000:
        #    break
    fout.close()
    meanOut.close()
    for w in oovdict:
        print 'oov',w,oovdict[w]





def generate_w2v_for_ivector_extractor():
    outputPath='./data/ivector_extractor_train'
    os.system('mkdir -p '+outputPath)
    w2vfeats=outputPath+'/w2vFeatures.ark'
    w2vmodel=models.Word2Vec.load(configure.IMDB_W2V_MODEL)
    rewrite_w2v_2_kaldi(configure.IMDB_FILE_TRAIN_IVECTOR_LINE_SENTANCE, w2vfeats, w2vmodel, lineID_Prefix='w2v', haslabel=False)





def rewrite2wordlist(inputFileName,outputFileName,haslabel=True,maxLine=-1):
    fin=codecs.open(inputFileName,'r','utf-8')
    fout=codecs.open(outputFileName,'w','utf-8')
    if haslabel:
        foutlb=codecs.open(outputFileName+u'.lb','w','utf-8')
    import jieba
    lcount=0
    while True:
        line=fin.readline()
        if not line:break
        line=line.strip()
        sentance=line
        if haslabel:
            lb=line.split(u' ')[0]
            sentance= u' '.join(line.split(u' ')[1:])

        while u'  ' in sentance:
            sentance=sentance.replace(u'  ', u' ')
        if sentance.endswith(u'None'): continue
        if len(sentance)<4:continue
        fout.write(lb+u' '+u' '.join(jieba.lcut(sentance)) + u'\n')
        if haslabel:
            foutlb.write(lb+u'\n')
        lcount+=1
        if lcount%5000==0:
            print lcount
        if maxLine!=-1:
            if lcount > maxLine:
                break
    fin.close()
    fout.close()
    if haslabel:
        foutlb.close()



def generate_train_for_w2v_vectors():
    outputPath=configure.KALDI_DATA_IMDB_TRAIN
    os.system('mkdir -p '+outputPath)
    w2vfeats=outputPath+'/w2vFeatures.ark'
    linesentance=configure.IMDB_FILE_TRAIN_W2V_LINE_SENTANCE
    #rewrite2wordlist(configure.TRAIN_SET,linesentance,haslabel=True,maxLine=500000)
    w2vmodel=models.Word2Vec.load(configure.IMDB_W2V_MODEL)
    rewrite_w2v_2_kaldi(linesentance,w2vfeats,w2vmodel,lineID_Prefix='train',haslabel=True)


def generate_semeval_for_w2v_vectors(linesentance,outputPath,lineID_Prefix='train'):
    #outputPath='./data/semeval_ivectors/'
    os.system('mkdir -p '+outputPath)
    w2vfeats=outputPath+'/w2vFeatures.ark'
    #linesentance=configure.SEMEVAL_FILE_LINE_SENTANCE
    w2vmodel=models.Word2Vec.load(configure.IMDB_W2V_MODEL)
    rewrite_w2v_2_kaldi(linesentance,w2vfeats,w2vmodel,lineID_Prefix=lineID_Prefix,haslabel=True)




def Covert2KaldiData():
    #generate_w2v_for_ivector_extractor()
    t=multiprocessing.Process(target=generate_w2v_for_ivector_extractor)
    t.daemon=False
    t.start()



    #generate_train_for_w2v_vectors()
    t1=multiprocessing.Process(target=generate_train_for_w2v_vectors)
    t1.daemon=False
    t1.start()

    #generate_semeval_for_w2v_vectors(configure.SEMEVAL_TRAIN_FILE_LINE_SENTANCE,'./data/semeval_train','semtrain')
    t2=multiprocessing.Process(target=generate_semeval_for_w2v_vectors,args=(configure.SEMEVAL_TRAIN_FILE_LINE_SENTANCE,configure.KALDI_DATA_SEMEVAL_TRAIN,'semtrain'))
    t2.daemon=False
    t2.start()


    t3=multiprocessing.Process(target=generate_semeval_for_w2v_vectors,args=(configure.SEMEVAL_DEV_FILE_LINE_SENTANCE,configure.KALDI_DATA_SEMEVAL_DEV,'semdev'))
    t3.daemon=False
    t3.start()


    t4=multiprocessing.Process(target=generate_semeval_for_w2v_vectors,args=(configure.SEMEVAL_DEV_TEST_FILE_LINE_SENTANCE,configure.KALDI_DATA_SEMEVAL_DEVTEST,'semdevtest'))
    t4.daemon=False
    t4.start()


#Covert2KaldiData()

def generate_kaggle_test_for_w2v_vectors(linesentance,outputPath,lineID_Prefix='train'):
    #outputPath='./data/semeval_ivectors/'
    os.system('mkdir -p '+outputPath)
    w2vfeats=outputPath+'/w2vFeatures.ark'
    #linesentance=configure.SEMEVAL_FILE_LINE_SENTANCE
    w2vmodel=models.Word2Vec.load(configure.IMDB_W2V_MODEL)
    rewrite_w2v_2_kaldi(linesentance,w2vfeats,w2vmodel,lineID_Prefix=lineID_Prefix,haslabel=True)



def Convert2KaldiData():
    #convert labeled train + unlabeled data for i

    generate_w2v_for_ivector_extractor()
    generate_train_for_w2v_vectors()
    generate_kaggle_test_for_w2v_vectors(configure.IMDB_FILE_TEST_LINE_SENTANCE, configure.KALDI_DATA_IMDB_TEST, 'ktest')
    generate_semeval_for_w2v_vectors(configure.SEMEVAL_TRAIN_FILE_LINE_SENTANCE,configure.KALDI_DATA_SEMEVAL_TRAIN,'semtrain')
    generate_semeval_for_w2v_vectors(configure.SEMEVAL_DEV_FILE_LINE_SENTANCE,configure.KALDI_DATA_SEMEVAL_DEV,'semdev')
    generate_semeval_for_w2v_vectors(configure.SEMEVAL_DEV_TEST_FILE_LINE_SENTANCE,configure.KALDI_DATA_SEMEVAL_DEVTEST,'semdevtest')



Convert2KaldiData()
