#coding=utf8
from tweetokenize import Tokenizer
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
import configure
import semeval2016libs

#Load data , train , dev, devtest

ids,trainy,trainx=semeval2016libs.load_semeval_text_only(configure.SUBTASK_A_TRAIN_FILE)
idsdev,evaly,evalx=semeval2016libs.load_semeval_text_only(configure.SUBTASK_A_TRAIN_DEV)
idsdevtest,evaly2,evalx2=semeval2016libs.load_semeval_text_only(configure.SUBTASK_A_TRAIN_DEVTEST)


#load data test

ids,testy,testx=semeval2016libs.load_semeval_text_only(configure.SUBTASK_A_TEST)


gettokens = Tokenizer()
tfv = TfidfVectorizer(min_df=2,use_idf=1,smooth_idf=1,ngram_range=(1,10),analyzer=gettokens.tokenize) #,stop_words='english')


trainx=tfv.fit_transform(trainx)
rawevalx=evalx
evalx=tfv.transform(evalx)
print tfv.get_feature_names()
print trainx.shape,evalx.shape


tsvd=TruncatedSVD(n_components=600,random_state=2016)   # this gives similar results as to Semeval , try n_components=600
trainx=tsvd.fit_transform(trainx)
evalx=tsvd.transform(evalx)


clf=LinearDiscriminantAnalysis()


clf.fit(trainx,trainy)
predictValue=clf.predict(evalx)



print semeval2016libs.scoreit(idsdev,predictValue,configure.SCORE_REF_DEV)

evalx2=tfv.transform(evalx2)
evalx2=tsvd.transform(evalx2)
predictValue=clf.predict(evalx2)


print semeval2016libs.scoreit(idsdevtest,predictValue,configure.SCORE_REF_DEVTEST)

testx=tfv.transform(testx)
testx=tsvd.transform(testx)
predictValue=clf.predict(testx)
print predictValue.shape
with open('results/scoring_test_data.sentence-three-point.subtaskA.pred.txt','w') as fout:
    for lid,pvalue in zip(ids,predictValue):
        fout.write('%s\t%s\n'%(lid,pvalue))

print semeval2016libs.scoreit(ids,predictValue,configure.SCORE_REF_TEST,score_script=configure.SCORE_SCRIPT_TEST)


