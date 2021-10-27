#!/bin/bash

set -eux

which wget
which gunzip
which unzip
which tar

pushd ./kytea
wget "http://www.phontron.com/kytea/download/model/jp-0.4.7-6.mod.gz"
gunzip "./jp-0.4.7-6.mod.gz"
popd
pushd ./sudachi
wget "http://sudachi.s3-website-ap-northeast-1.amazonaws.com/sudachidict/sudachi-dictionary-20210802-core.zip"
unzip "./sudachi-dictionary-20210802-core.zip"
popd
pushd ./sudachi.rs
./fetch_dictionary.sh
popd

wget "http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz"
tar xf "./kftt-data-1.0.tar.gz"
