#System 1 for Aicyber


##A vector space model approach.

##Preprocessing

The text data is first being processed by tweet tokenizer, emoticons are preserved as tokens, some emoticons are replaced by words , eg :) -> good , :( -> bad

##Features

Bag-of-ngram feature is extracted and filtered by a TF-IDF (Salton, 1991) selection.

Resulting feature dimension is around 3800, it is then reduced to 400 by truncated singular value decomposition (SVD) (Klema and Laub, 1980; Halko et al., 2009).

This process is also known as Latent Semantic Analysis (LSA) or Vector Space Model (Turney and Pantel, 2010).

##Classifier

Finally a Linear Discriminant Analysis (LDA) classifier (Hastie et al., 2009) is trained to classify the test data.

##Training and verification data

The SemEval 2016 training dataset which contains 3887 tweets are selected to train the TF-IDF,SVD and LDA. Development dataset is used for tuning parameters and develop-test dataset are used for
local testing.

##Results:

###SVD DIM=400
Twitter2015 0.4617  Twitter2016 0.4048
###SVD DIM=600
Twitter2015 0.4732  Twitter2016 0.4261
