#!/bin/bash
# Copyright 2015   David Snyder
#           2015   Johns Hopkins University (Author: Daniel Garcia-Romero)
#           2015   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
#trials_female=data/sre10_test_female/trials
#trials_male=data/sre10_test_male/trials
#trials=data/sre10_test/trials
num_components=2048 # Larger than this doesn't make much of a difference.

# Prepare the SRE 2010 evaluation data.
#default is 3
ivector_extractor_trainDir=ivector_extractor_train
deltaWindows=3
vectorSize=200
maxdataCountDiag=32000
maxdataCount=64000

preparedata=true;
trainDGMM=true;
trainiVectorExtractor=true;
extractIvectorIMDB=true;
extractIvectorSemeval=true;



if $preparedata;then

#for name in ivector_extractor_train; do
  utils/fix_data_dir.sh data/${ivector_extractor_trainDir}
  # Reduce the amount of training data for the UBM.
    utils/subset_data_dir.sh data/${ivector_extractor_trainDir} $maxdataCountDiag data/train_16k
    utils/subset_data_dir.sh data/${ivector_extractor_trainDir} $maxdataCount data/train_32k
#done

fi

if $trainiVectorExtractor;then
    exptag=NGMM_${num_components}_W_${deltaWindows}_DIM_${vectorSize}
# Train UBM and i-vector extractor.
        if $trainDGMM;then

        sid/train_diag_ubm.sh --nj 4 --delta-window $deltaWindows --cmd "$train_cmd -l mem_free=20G,ram_free=20G"\
            data/train_16k $num_components \
            exp/diag_ubm_$exptag

            sid/train_full_ubm.sh --nj 4 --remove-low-count-gaussians false  \
            --cmd "$train_cmd -l mem_free=25G,ram_free=25G" data/train_32k \
            exp/diag_ubm_$exptag exp/full_ubm_$exptag
        fi
    sid/train_ivector_extractor.sh --cmd "$train_cmd -l mem_free=35G,ram_free=35G" \
      --ivector-dim $vectorSize --nj 4  \
      --num-iters 5 exp/full_ubm_$exptag/final.ubm data/${ivector_extractor_trainDir} \
      exp/extractor_$exptag

fi


if $extractIvectorIMDB;then
    exptag=NGMM_${num_components}_W_${deltaWindows}_DIM_${vectorSize}

    for datafolder in imdb_train imdb_test
    do
        outputpath=exp/ivectors_${datafolder}_$exptag
        # Extract i-vectors.
        #extractor is exp/extractor3/final.ie
        extractorFolder=exp/extractor_$exptag
        sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=6G,ram_free=6G" --nj 5 \
           $extractorFolder data/$datafolder \
           $outputpath

        copy-feats scp:$outputpath/ivector.scp ark,t: > $outputpath/feats.txt
    done
fi



if $extractIvectorSemeval;then
    exptag=NGMM_${num_components}_W_${deltaWindows}_DIM_${vectorSize}

    for datafolder in semeval_train semeval_dev semeval_devtest
    do
        outputpath=exp/ivectors_${expdata}_$exptag
        # Extract i-vectors.
        #extractor is exp/extractor3/final.ie
        extractorFolder=exp/extractor_$exptag
        sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=6G,ram_free=6G" --nj 5 \
           $extractorFolder data/$datafolder \
           $outputpath
        copy-feats scp:$outputpath/ivector.scp ark,t: > $outputpath/feats.txt
    done

fi


exit 0
