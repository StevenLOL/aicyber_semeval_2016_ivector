from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import imdb_bag_of_word_libs
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tweetokenize import Tokenizer
gettokens = Tokenizer()
train,test,unlabeled_train=imdb_bag_of_word_libs.loadUnlabedData()

def mytokenlizer(input):
    input=input.lower()  #without this 0.4024 with this 0.4185
    rv=gettokens.tokenize(input)
    return rv

print type(train),train.shape

trainx=train['review'].values
trainy=train['sentiment'].values
testx=test['review'].values

trainx=[s[1:-1] for s in trainx]
testx=[s[1:-1] for s in testx]

print type(trainx),trainx[0]
trainx=np.array(trainx)
testx=np.array(testx)


skf =KFold(len(trainy),n_folds=4,random_state=2016) #StratifiedKFold(trainy, n_folds=4)


def processTraining(cvtrainx,cvtrainy,cvevalx,prob=False):
    print cvtrainx[0]
    #cvevalx=[' '.join(s) for s in cvevalx]
    print cvevalx[0]
    tfv = TfidfVectorizer(min_df=10,  max_features=None,
        strip_accents='unicode', analyzer=mytokenlizer,
        ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')

    cvtrainx=tfv.fit_transform(cvtrainx)
    cvevalx=tfv.transform(cvevalx)
    tsvd=TruncatedSVD(n_components=600,random_state=2016)
    cvtrainx=tsvd.fit_transform(cvtrainx)
    cvevalx=tsvd.transform(cvevalx)
    print len(tfv.get_feature_names())
    print tfv.get_feature_names()[0:10]
    clf=LinearDiscriminantAnalysis()
    clf.fit(cvtrainx,cvtrainy)
    if prob:
        predictValue=clf.predict_proba(cvevalx)
    else:
        predictValue=clf.predict(cvevalx)
    return predictValue

for train_index, test_index in skf:
    break;
    cvtrainx=trainx[train_index]
    cvtrainy=trainy[train_index]
    cvevalx=trainx[test_index]
    cvevaly=trainy[test_index]

    predictValue=processTraining(cvtrainx,cvtrainy,cvevalx)
    print confusion_matrix(cvevaly,predictValue)
    print  classification_report(cvevaly,predictValue)
    print '\n'


print 'train on full data'

testid=test['id'].values
testid=[s.replace('"','') for s in testid]
trueid=list()
print trainy[0:10]
for t in testid:
    intid=int(t.split('_')[1])
    if intid>=5:
        trueid.append(1)
    else:
        trueid.append(0)
print 'testid[0]',testid[0]
predictValue=processTraining(trainx,trainy,testx,prob=False)
print confusion_matrix(trueid,predictValue)
print classification_report(trueid,predictValue,digits=4)
from sklearn.metrics import accuracy_score
print accuracy_score(trueid,predictValue)











'''
[[10906  1594]
 [ 1310 11190]]
             precision    recall  f1-score   support

          0     0.8928    0.8725    0.8825     12500
          1     0.8753    0.8952    0.8851     12500

avg / total     0.8840    0.8838    0.8838     25000

0.88384
'''