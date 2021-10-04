#!/bin/bash

set -eux

INPUT_DATA="./kftt-data-1.0/data/orig/kyoto-train.ja"

for i in 0 1 2 3 4 5 6 7 8 9
do
    for j in 0 1 2 3 4 5 6 7 8 9
    do
        echo "iter" $i $j

        ./kytea/src/bin/kytea -model "./kytea/jp-0.4.7-6.mod" -notags < $INPUT_DATA > /dev/null

        ../target/release/predict --model "../jp-0.4.7-6.tokenize.mod" < $INPUT_DATA > /dev/null

        ./mecab/tmpusr/bin/mecab -Owakati < $INPUT_DATA > /dev/null

        pushd ./kuromoji
        mvn exec:java -Dexec.mainClass=kuromoji_bench.App < ../$INPUT_DATA > /dev/null
        popd

        ./lindera/target/release/lindera -O wakati < $INPUT_DATA > /dev/null

        pushd ./sudachi
        mvn exec:java -Dexec.mainClass=sudachi_bench.App < ../$INPUT_DATA > /dev/null
        popd

        ./sudachi.rs/target/release/sudachi -w -m C < $INPUT_DATA > /dev/null
    done
done
