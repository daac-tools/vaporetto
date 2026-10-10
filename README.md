# 🛥 Vaporetto: Very accelerated pointwise prediction based tokenizer

Vaporetto is a fast and lightweight pointwise prediction-based tokenizer.
This repository includes both a Rust crate that provides APIs for Vaporetto and CLI frontends.

[![Crates.io](https://img.shields.io/crates/v/vaporetto)](https://crates.io/crates/vaporetto)
[![Documentation](https://docs.rs/vaporetto/badge.svg)](https://docs.rs/vaporetto)
[![Build Status](https://github.com/daac-tools/vaporetto/actions/workflows/rust.yml/badge.svg)](https://github.com/daac-tools/vaporetto/actions)
[![Slack](https://img.shields.io/badge/join-chat-brightgreen?logo=slack)](https://join.slack.com/t/daac-tools/shared_invite/zt-1pwwqbcz4-KxL95Nam9VinpPlzUpEGyA)

[日本語のドキュメント](README-ja.md)

[Wasm Demo](https://vaporetto-demo.pages.dev/) (takes a little time to load the model.)

A Python wrapper is also available [here](https://github.com/daac-tools/python-vaporetto).

## Example Usage

### Try Word Segmentation

This software is implemented in Rust. Install `rustc` and `cargo` following [the documentation](https://www.rust-lang.org/tools/install) beforehand.

Vaporetto provides three ways to generate tokenization models:

#### Download Distribution Model

The first is the simplest way, which is to download a model we have trained.
Models are available [here](https://github.com/daac-tools/vaporetto-models/releases).

We chose `bccwj-suw+unidic_pos+pron`:
```
% wget https://github.com/daac-tools/vaporetto-models/releases/download/v0.5.0/bccwj-suw+unidic_pos+pron.tar.xz
```

Each file is a compressed file containing a model file and license terms, so you need to decompress the downloaded file as shown in the following command:
```
% tar xf ./bccwj-suw+unidic_pos+pron.tar.xz
```

The distribution models were created by Vaporetto 0.6 or earlier, whose model format is not compatible with the current version.
Convert the model into the current format using the following command:
```
% cargo run --release -p convert_bincode_model -- --model-in ./bccwj-suw+unidic_pos+pron/bccwj-suw+unidic_pos+pron.model.zst --model-out path/to/bccwj-suw+unidic_pos+pron.model.zst
```

To perform tokenization, run the following command:
```
% echo 'ヴェネツィアはイタリアにあります。' | cargo run --release -p predict -- --model path/to/bccwj-suw+unidic_pos+pron.model.zst
```

The following will be output:
```
ヴェネツィア は イタリア に あり ます 。
```

##### Notes for Vaporetto APIs

The distribution models are compressed in the zstd format.
If you want to load these compressed models with the *vaporetto* API,
you must decompress them outside of the API.

```rust
// Requires zstd crate or ruzstd crate
let reader = zstd::Decoder::new(File::open("path/to/model.zst")?)?;
let model = Model::read(reader)?;
```

You can also decompress the file using the *unzstd* command, which is bundled with modern Linux
distributions.

#### Convert KyTea's Model

The second is also a simple way, which is to convert a model trained by KyTea.
First of all, download the model of your choice from the [KyTea Models](http://www.phontron.com/kytea/model.html) page.

We chose `jp-0.4.7-5.mod.gz`:
```
% wget http://www.phontron.com/kytea/download/model/jp-0.4.7-5.mod.gz
```

Each file is a compressed file, so you need to decompress the downloaded model file as shown in the following command:
```
% gunzip ./jp-0.4.7-5.mod.gz
```

To convert a KyTea model into a Vaporetto model, run the following command in the Vaporetto root directory.
```
% cargo run --release -p convert_kytea_model -- --model-in path/to/jp-0.4.7-5.mod --model-out path/to/jp-0.4.7-5-tokenize.model.zst
```

Now you can perform tokenization. Run the following command:
```
% echo 'ヴェネツィアはイタリアにあります。' | cargo run --release -p predict -- --model path/to/jp-0.4.7-5-tokenize.model.zst
```

The following will be output:
```
ヴェネツィア は イタリア に あ り ま す 。
```

#### Train Your Model

The third way, which is mainly for researchers, is to prepare a training corpus and train your tokenization models.

Vaporetto can train from two types of corpora: fully annotated corpora and partially annotated corpora.

Fully annotated corpora are corpora in which all character boundaries are annotated with either token boundaries or internal positions of tokens.
This is the data in the form of spaces inserted into the boundaries of the tokens, as shown below:

```
ヴェネツィア は イタリア に あり ます 。
火星 猫 の 生態 の 調査 結果
```

Besides, partially annotated corpora are corpora in which only some character boundaries are annotated.
Each character boundary is annotated in the form of `|` (token boundary), `-` (not token boundary), and ` ` (unknown).
Here is an example:

```
ヴ-ェ-ネ-ツ-ィ-ア|は|イ-タ-リ-ア|に|あ り ま す|。
火-星 猫|の|生-態|の|調-査 結-果
```

To train a model, use the following command:

```
% cargo run --release -p train -- --model ./your.model.zst --tok path/to/full.txt --part path/to/part.txt --dict path/to/dict.txt --solver 5
```

The `--tok` argument specifies a fully annotated corpus, and the `--part` argument specifies a partially annotated corpus.
You can also specify a word dictionary with the `--dict` argument.
A word dictionary is a file that lists words line by line and can be tagged as needed:

```
トスカーナ
パンツァーノ
灯里/名詞-固有名詞-人名-名/アカリ
形態/名詞-普通名詞-一般/ケータイ
```

The trainer does not accept empty lines.
Therefore, remove all empty lines from the corpus before training.

You can specify all arguments above multiple times.

### Model Manipulation

Sometimes, your model will output different results than what you expect.
For example, `外国人参政権` is split into wrong tokens in the following command.
We use the `--scores` option to show the score of each character boundary:
```
% echo '外国人参政権と政権交代' | cargo run --release -p predict -- --scores --model path/to/bccwj-suw+unidic_pos+pron.model.zst
外国 人 参 政権 と 政権 交代
0:外国 -10784
1:国人 17935
2:人参 5308
3:参政 3833
4:政権 -3299
5:権と 14635
6:と政 17653
7:政権 -12705
8:権交 11611
9:交代 -5794
```

The correct is `外国 人 参政 権`.
To split `外国人参政権` into correct tokens, manipulate the model in the following steps so that the sign of score of `参政権` becomes inverted:

1. Dump a dictionary by the following command:
   ```
   % cargo run --release -p manipulate_model -- --model-in path/to/bccwj-suw+unidic_pos+pron.model.zst --dump-dict path/to/dictionary.csv
   ```

2. Edit the dictionary.

   The dictionary is a CSV file. Each row contains a string pattern, a corresponding weight array, and a comment in the following order:

   * `word` - A string pattern (usually a word)
   * `weights` - A weight array. When the input string contains the pattern, these weights are added to the character boundaries of the range of the pattern found.
   * `comment` - A comment that does not affect the behavior.

   Vaporetto splits a text when the total weight of the boundary is a positive number, so we add a new entry as follows:
   ```diff
    参撾,3328 -5545 3514,
    参政,3328 -5545 3514,
   +参政権,0 -10000 10000 0,参政/権
    参朝,3328 -5545 3514,
    参校,3328 -5545 3514,
   ```

   In this case, `-10000` will be added between `参` and `政`, and `10000` will be added between `政` and `権`.
   Because `0` is specified at both ends of the pattern, no scores are added at those positions.

   Note that Vaporetto uses 32-bit integers for the total weight, so you have to be careful about overflow.

   In addition, The dictionary cannot contain duplicated words.
   When the dictionary already contains the word, you have to edit existing weights.

3. Replaces weight data of a model file
   ```
   % cargo run --release -p manipulate_model -- --model-in path/to/bccwj-suw+unidic_pos+pron.model.zst --replace-dict path/to/dictionary.csv --model-out path/to/bccwj-suw+unidic_pos+pron-new.model.zst
   ```

Now `外国人参政権` is split into correct tokens.
```
% echo '外国人参政権と政権交代' | cargo run --release -p predict -- --scores --model path/to/bccwj-suw+unidic_pos+pron-new.model.zst
外国 人 参政 権 と 政権 交代
0:外国 -10784
1:国人 17935
2:人参 5308
3:参政 -6167
4:政権 6701
5:権と 14635
6:と政 17653
7:政権 -12705
8:権交 11611
9:交代 -5794
```

### Tag prediction

Vaporetto experimentally supports tagging (e.g., part-of-speech and pronunciation tags).

To train tags, add slashes and tags following each token in the dataset as follows:

* For fully annotated corpora
  ```
  この/連体詞/コノ 人/名詞/ヒト は/助詞/ワ 火星/名詞/カセイ 人/接尾辞/ジン です/助動詞/デス
  ```

* For partially annotated corpora
  ```
  ヴ-ェ-ネ-ツ-ィ-ア/名詞|は/助詞|イ-タ-リ-ア/名詞|に/助詞|あ-り ま-す
  ```

You can also specify tag information to dictionaries as well as corpora.
When the predictor cannot predict a tag using the model, the tag specified in the dictionary will be annotated to the token.

If the dataset contains tags, the `train` command automatically trains them.

In prediction, tags are not predicted by default, so you have to specify the `--predict-tags` argument to the `predict` command if necessary.

If you specify the `--tag-scores` argument, the score of each candidate calculated during tag prediction is displayed.
If there is only one candidate, the score becomes 0.

```
% echo "花が咲く" | cargo run --release -p predict -- --model path/to/bccwj-suw+unidic_pos+pron.model.zst --predict-tags --tag-scores
花/名詞-普通名詞-一般/ハナ が/助詞-格助詞/ガ 咲く/動詞-一般/サク
花	名詞-普通名詞-一般:18613,接尾辞-名詞的-一般:-18613	ハナ:19973,バナ:-20377,カ:-20480,ゲ:-20410
が	助詞-接続助詞:-20408,助詞-格助詞:23543,接続詞:-25332	ガ:0
咲く	動詞-一般:0	サク:0
```

## Speed Comparison of Various Tokenizers

Vaporetto is 8.7 times faster than KyTea.

Details can be found [here](https://github.com/daac-tools/vaporetto/wiki/Speed-Comparison).

![](./figures/comparison.svg)

## Slack

We have a Slack workspace for developers and users to ask questions and discuss a variety of topics.

 * https://daac-tools.slack.com/
 * Please get an invitation from [here](https://join.slack.com/t/daac-tools/shared_invite/zt-1pwwqbcz4-KxL95Nam9VinpPlzUpEGyA).

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

See [the guidelines](./CONTRIBUTING.md).

## References

Technical details of the whole tokenizer system:

 * Koichi Akabe, Shunsuke Kanda, Yusuke Oda, Shinsuke Mori. [Vaporetto: Efficient Japanese Tokenization Based on Improved Pointwise Linear Classification](https://arxiv.org/abs/2406.17185). arXiv. 2024.

 * 赤部 晃一, 神田 峻介, 小田 悠介, 森 信介. [Vaporetto: 点予測法に基づく高速な日本語トークナイザ](https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/D2-5.pdf). 言語処理学会第28回年次大会 (NLP2022). 浜松. 2022年3月. (in Japanese)

 * [Blog post](https://tech.legalforce.co.jp/entry/2021/09/28/180844) (in Japanese)

Technical details of DAACs, the pattern match algorithm underlying the tokenizer system:

 * Shunsuke Kanda, Koichi Akabe, and Yusuke Oda. [Engineering faster double-array Aho-Corasick automata](https://doi.org/10.1002/spe.3190). Software: Practice and Experience (SPE), 53(6): 1332–1361, 2023 ([arXiv](https://arxiv.org/abs/2207.13870))
