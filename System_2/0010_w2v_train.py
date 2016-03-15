#coding=utf-8

from gensim.models.word2vec import LineSentence
from gensim.models import word2vec,Word2Vec
from gensim import corpora, models
import logging
import codecs
import configure


def trainW2V(sentences,model_name = "300features_40minwords_10context.bin"):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 20    # Word vector dimensionality
    min_word_count = 10 #25   # Minimum word count
    num_workers = 8       # Number of threads to run in parallel
    context = 7          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)x
    from gensim.models import word2vec
    print "Training model..."
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()

    model.save(model_name)

datalist= LineSentence(configure.IMDB_FILE_ALL_LINE_SENTANCE)  #loadGenSimTrainData('aicyber_wiki_weibo_words.txt')
#print len(datalist)
#print datalist[0]
#print datalist[1]
#print len(datalist)
model_name = configure.IMDB_W2V_MODEL
trainW2V(datalist,model_name = model_name)

#testmodel
#model_name = "daiwei_wiki_weibo_20w_jieba.bin"
model =models.Word2Vec.load(model_name)
parts= model.most_similar('good')
for p in parts:
    print p[0],p[1]
print model.syn0.shape
