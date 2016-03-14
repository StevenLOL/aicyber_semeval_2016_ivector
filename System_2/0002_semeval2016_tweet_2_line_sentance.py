#coding=utf-8
import configure
from imdb_bag_of_word_libs import *
#x=loadData(configure.KAGLE_BOW_DATA_FOLD+'/')



def rewrtieSemeval(fname,targetFilename):
    trainx=list()
    #train = pd.read_csv( os.path.join(configure.KAGLE_BOW_DATA_FOLD,"/home/steven/Downloads/training.1600000.processed.noemoticon.csv"), header=0,
     #   delimiter=",", quoting=1,encoding='utf-8' )
    fdata=codecs.open(fname,encoding='utf-8').readlines()
    trainx=[s.split(u'\t')[2].strip() for s in fdata]
    trainy=[s.split(u'\t')[1].strip() for s in fdata]
    #ldict={u'neutral':1,u'negative':0,u'positive':2}
    #trainy=[unicode(ldict[s]) for s in trainy]
    print type(trainx),type(trainx[0])
    #trainx=[s.replace(u';D',u':D') for s in trainx]
    trainx=map(processLine,trainx)
    with codecs.open(targetFilename,'w',encoding='utf-8') as fout:
        for lb, line in zip(trainy, trainx):
            if len(line)>2:
                fout.write(lb+u' '+u' '.join(line).strip()+u'\n')
   # def saveStandFord():






rewrtieSemeval(configure.SEMEVAL_TRAIN,configure.SEMEVAL_TRAIN_FILE_LINE_SENTANCE)
rewrtieSemeval(configure.SEMEVAL_DEV,configure.SEMEVAL_DEV_FILE_LINE_SENTANCE)
rewrtieSemeval(configure.SEMEVAL_DEV_TEST,configure.SEMEVAL_DEV_TEST_FILE_LINE_SENTANCE)
