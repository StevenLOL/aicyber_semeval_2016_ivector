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
import imdb_bag_of_word_libs
import numpy as np
ts=time.time()

devtest='./exp/ivectors_semeval_devtest_NGMM_2048_W_2_DIM_200/feats.txt'
dev='./exp/ivectors_semeval_dev_NGMM_2048_W_2_DIM_200/feats.txt'
train='./exp/ivectors_semeval_train_NGMM_2048_W_2_DIM_200/feats.txt'



trainy,trainx=imdb_bag_of_word_libs.loadFeatsText(train)
trainy=imdb_bag_of_word_libs.kaldiID_2_LB(trainy)
evaly,evalx=imdb_bag_of_word_libs.loadFeatsText(dev)
evaly=imdb_bag_of_word_libs.kaldiID_2_LB(evaly)

evaly2,evalx2=imdb_bag_of_word_libs.loadFeatsText(devtest)
evaly2=imdb_bag_of_word_libs.kaldiID_2_LB(evaly2)


robust_scaler = RobustScaler()
trainx=robust_scaler.fit_transform(trainx)
evalx=robust_scaler.transform(evalx)

clf= LinearDiscriminantAnalysis() #
clf.fit(trainx,trainy)
predictValue=clf.predict(evalx)

print semeval2016_libs.scoreSameOrder(predictValue,configure.SCORE_REF_DEV)

evalx2=robust_scaler.transform(evalx2)
predictValue=clf.predict(evalx2)


print semeval2016_libs.scoreSameOrder(predictValue,configure.SCORE_REF_DEVTEST)
