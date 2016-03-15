#coding=utf-8
from multiprocessing import Pool

import configure
import codecs
import imdb_bag_of_word_libs
import pandas as pd
import os
#x=loadData(configure.KAGLE_BOW_DATA_FOLD+'/')
def loadUnlabedData():

    test = pd.read_csv(configure.IMDB_FILE_TEST, header=0, delimiter="\t", quoting=3, encoding='utf-8')


    return test


test=loadUnlabedData()
testx=test['review'].values

#trainx,testx,unlabeled=map(unicode,trainx),map(unicode,testx),map(unicode,unlabeled)

ids=test['id'].values

testx=[s[1:-1] for s in testx]
ids=[s[1:-1] for s in ids]
pool=Pool(processes=3)

testx=pool.map(imdb_bag_of_word_libs.processLine,testx)


#save train and unlabed
#save train to eval set



def saveTrain():
    with codecs.open(configure.IMDB_FILE_TEST_LINE_SENTANCE, 'w', encoding='utf-8') as fout:
        for lb,line in zip(ids, testx):
            fout.write(unicode(lb)+u' '+u' '.join(line).strip()+u'\n')









saveTrain()

