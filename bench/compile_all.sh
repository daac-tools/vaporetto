#!/bin/bash

set -eux

which patch
which cargo
which autoreconf
which libtool
which make
which mvn

set +e

patch -p1 -N < ./elapsed_time.patch

set -e

pushd ..
cargo build --release
./target/release/convert_kytea_model --model-in "./bench/kytea/jp-0.4.7-6.mod" --model-out "./jp-0.4.7-6.tokenize.mod"
popd

pushd ./kytea
autoreconf -i
./configure
make
popd

pushd ./mecab/mecab
./configure --prefix=$(cd .. && pwd)/tmpusr
make
make install
popd
pushd ./mecab/mecab-ipadic
./configure --with-charset=utf8 --prefix=$(cd .. && pwd)/tmpusr --with-mecab-config=../mecab/mecab-config
make
make install
popd

pushd ./kuromoji
mvn compile
popd

pushd ./lindera
cargo build --release
popd

pushd ./sudachi
mvn compile
popd

pushd ./sudachi.rs
cargo build --release
popd
