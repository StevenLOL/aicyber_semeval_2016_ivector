from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold
import sklearn
import configure
import time
import numpy as np
ts=time.time()
#y,x=configure.loadFeatsText('/home/steven/gits/development_nlp_w2v_ivector/exp/ivectors_sre10_train/feats.txt')

outputfile='./tempfolder/results_w2v_mean.csv'
y,x=configure.loadFeatsText('./exp/ivectors_imdb_train_NGMM_2048_W_2_DIM_200/feats.txt')
#y2,x2=configure.loadFeatsText('./data/train_ivectors/w2vFeatures.ark.mean')
print y[0],len(x[0]),x[0]
#[ s1.extend(s2) for s1,s2 in zip(x,x2)]
#x=x2
print y[0],len(x[0]),x[0]
testy,testx=configure.loadFeatsText('./exp/ivectors_imdb_test_NGMM_2048_W_2_DIM_200/feats.txt')
#testy2,testx2=configure.loadFeatsText('./data/kaggle_test/w2vFeatures.ark.mean')
#[ s1.extend(s2) for s1,s2 in zip(testx,testx2)]
#testx=testx2
#assert y==y2
print 'done in',time.time()-ts,len(x),len(y)

y=configure.kaldiID_2_LB(y)
print y[0],x[0]
#testy=configure.kaldiID_2_LB(testy)




x=np.array(x)
y=np.array(y)



trainx,trainy=x,y

robust_scaler = RobustScaler()
trainx=robust_scaler.fit_transform(trainx)
evalx=robust_scaler.transform(testx)
#clf= LinearSVC( penalty='l2',dual=False, tol=1e-3)
#clf= RandomForestClassifier(n_estimators=300,n_jobs=-1,random_state=2016) #
clf= LinearDiscriminantAnalysis() #
#clf=BaggingClassifier(base_estimator=SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"),n_estimators=5,n_jobs=-1)
clf.fit(trainx,trainy)
predictValue=clf.predict(evalx)

sdict=dict()
ptrue=list()
for id,score in zip(testy,predictValue):
    sdict[id]=score
    #print id,score
    truevalue=int(id.split('_')[2])
    if truevalue>=5:
        ptrue.append('1')
    else:
        ptrue.append('0')

print confusion_matrix(ptrue,predictValue)
print classification_report(ptrue,predictValue,digits=4)

from sklearn.metrics import accuracy_score
print accuracy_score(ptrue,predictValue)

