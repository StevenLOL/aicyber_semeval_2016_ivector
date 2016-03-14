#coding=utf-8
import configure
import codecs
import imdb_bag_of_word_libs
#x=loadData(configure.KAGLE_BOW_DATA_FOLD+'/')
train,test,unlabetrain=imdb_bag_of_word_libs.loadUnlabedData()
from multiprocessing import Pool
print type(train),type(test)

print type(train),train.shape

trainx=train['review'].values

trainy=train['sentiment'].values
testx=test['review'].values

unlabeled=unlabetrain['review']

#trainx,testx,unlabeled=map(unicode,trainx),map(unicode,testx),map(unicode,unlabeled)

print type(trainx),type(trainx[0]),trainx.shape,trainx[0]

trainx=[s[1:-1] for s in trainx]
unlabeled=[s[1:-1] for s in unlabeled]
testx=[s[1:-1] for s in testx]

pool=Pool(processes=3)
trainx=pool.map(imdb_bag_of_word_libs.processLine, trainx)

testx=pool.map(imdb_bag_of_word_libs.processLine, testx)

unlabeled=pool.map(imdb_bag_of_word_libs.processLine, unlabeled)



print type(trainx),type(trainx[0]),trainx[0]

#save train and unlabed
#save train to eval set

def saveTrainUnlabed():
    unlabeled.extend(trainx)
    with codecs.open(configure.KAGLE_FILE_LINE_SENTANCE,'w',encoding='utf-8') as fout:
        for line in unlabeled:
            fout.write(u' '.join(line).strip()+u'\n')

def saveTrain():
    with codecs.open(configure.KAGLE_FILE_TRAIN_W2V_LINE_SENTANCE, 'w', encoding='utf-8') as fout:
        for lb,line in zip(trainy, trainx):
            fout.write(unicode(lb)+u' '+u' '.join(line).strip()+u'\n')








saveTrainUnlabed()
saveTrain()

