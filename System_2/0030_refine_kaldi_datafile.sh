#!/usr/bin/env bash
# after convert all line sentance to kaldi ark format
# this script will then convert kaldi ark to ark and scp format, and generate a dummy utt2spk file though this file is not used in the training
#./data/*
for x in ./data/*
do
    echo $x
    ./conver_feats.sh $x &
done
wait
echo "Done!"
echo "Now change setting in the run.w2v.sh file and start to train iVector extracor"

