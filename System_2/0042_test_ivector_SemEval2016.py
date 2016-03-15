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
import semeval2016_libs
import numpy as np
ts=time.time()

devtest='./exp/ivectors_semeval_devtest_NGMM_2048_W_2_DIM_200/feats.txt'
dev='./exp/ivectors_semeval_dev_NGMM_2048_W_2_DIM_200/feats.txt'
train='./exp/ivectors_semeval_train_NGMM_2048_W_2_DIM_200/feats.txt'



trainy,trainx=configure.loadFeatsText(train)
trainy=configure.kaldiID_2_LB(trainy)
evaly,evalx=configure.loadFeatsText(dev)
evaly=configure.kaldiID_2_LB(evaly)

evaly2,evalx2=configure.loadFeatsText(devtest)
evaly2=configure.kaldiID_2_LB(evaly2)


robust_scaler = RobustScaler()
trainx=robust_scaler.fit_transform(trainx)
evalx=robust_scaler.transform(evalx)

clf= LinearDiscriminantAnalysis() #
clf.fit(trainx,trainy)
predictValue=clf.predict(evalx)
print confusion_matrix(evaly,predictValue)
print classification_report(evaly,predictValue)

print semeval2016_libs.scoreSameOrder(predictValue,configure.SCORE_REF_DEV)

evalx2=robust_scaler.transform(evalx2)
predictValue=clf.predict(evalx2)
print confusion_matrix(evaly2,predictValue)
print classification_report(evaly2,predictValue)

print semeval2016_libs.scoreSameOrder(predictValue,configure.SCORE_REF_DEVTEST)
