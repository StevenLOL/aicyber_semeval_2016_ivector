#!/usr/bin/env bash
#after convert sentances to sentance matrix via w2v
#this script will transform the data to kaldi format, to say in  ark (data)  and scp (index) pair

path=$1
. path.sh
copy-feats ark:$path/w2vFeatures.ark ark,scp:$path/feats.ark,$path/feats.scp

#after convertion , generate the utt2spk
python scp_2_utt2spk.py $path/feats.scp $path/utt2spk