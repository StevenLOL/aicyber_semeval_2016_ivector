from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import configure
import time
import numpy as np
import imdb_bag_of_word_libs
ts=time.time()

outputfile='./tempfolder/results_w2v_mean.csv'
y,x=imdb_bag_of_word_libs.loadFeatsText('./exp/ivectors_imdb_train_NGMM_2048_W_2_DIM_200/feats.txt')

print y[0],len(x[0]),x[0]

testy,testx=imdb_bag_of_word_libs.loadFeatsText('./exp/ivectors_imdb_test_NGMM_2048_W_2_DIM_200/feats.txt')

print 'done in',time.time()-ts,len(x),len(y)

y=imdb_bag_of_word_libs.kaldiID_2_LB(y)
print y[0],x[0]


x=np.array(x)
y=np.array(y)



trainx,trainy=x,y

robust_scaler = RobustScaler()
trainx=robust_scaler.fit_transform(trainx)
evalx=robust_scaler.transform(testx)
clf= LinearDiscriminantAnalysis()
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

