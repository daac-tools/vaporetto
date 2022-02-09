# 🛥 VAporetto: POintwise pREdicTion based TOkenizer

Vaporetto is a fast and lightweight pointwise prediction based tokenizer.
This repository includes both a Rust crate that provides APIs for Vaporetto and CLI frontends.

[![Crates.io](https://img.shields.io/crates/v/vaporetto)](https://crates.io/crates/vaporetto)
[![Documentation](https://docs.rs/vaporetto/badge.svg)](https://docs.rs/vaporetto)
![Build Status](https://github.com/legalforce-research/vaporetto/actions/workflows/rust.yml/badge.svg)

[Technical details](https://tech.legalforce.co.jp/entry/2021/09/28/180844) (Japanese)

## Example Usage

### Try Word Segmentation

This software is implemented in Rust. Install `rustc` and `cargo` following [the documentation](https://www.rust-lang.org/tools/install) beforehand.

Vaporetto provides two ways to generate tokenization models:

#### Convert KyTea's Model

The first is the simplest way, which is to convert a model that has been trained by KyTea.
First of all, download the model of your choice from the [KyTea Models](http://www.phontron.com/kytea/model.html) page.

We chose `jp-0.4.7-5.mod.gz`:
```
% wget http://www.phontron.com/kytea/download/model/jp-0.4.7-5.mod.gz
```

Each model is compressed, so you need to decompress the downloaded model file like the following command:
```
% gunzip ./jp-0.4.7-5.mod.gz
```

To convert a KyTea model into a Vaporetto model, run the following command in the Vaporetto root directory.
If necessary, the Rust code will be compiled before the conversion process.
```
% cargo run --release -p convert_kytea_model -- --model-in path/to/jp-0.4.7-5.mod --model-out path/to/jp-0.4.7-5-tokenize.model.zst
```

Now you can perform tokenization. Run the following command:
```
% echo '火星猫の生態の調査結果' | cargo run --release -p predict -- --model path/to/jp-0.4.7-5-tokenize.model.zst
```

The following will be output:
```
火星 猫 の 生態 の 調査 結果
```

#### Train Your Own Model

The second way, which is mainly for researchers, is to prepare your own training corpus and train your own tokenization models.

Vaporetto can train from two types of corpora: fully annotated corpora and partially annotated corpora.

Fully annotated corpora are corpora in which all character boundaries are annotated with either token boundaries or internal positions of tokens.
This is the data in the form of spaces inserted into the boundaries of the tokens, as shown below:

```
ヴェネツィア は イタリア に あ り ま す 。
火星 猫 の 生態 の 調査 結果
```

On the other hand, partially annotated corpora are corpora in which only some character boundaries are annotated.
Each character boundary is annotated in the form of `|` (token boundary), `-` (not token boundary), and ` ` (unknown).
Here is an example:

```
ヴ-ェ-ネ-ツ-ィ-ア|は|イ-タ-リ-ア|に|あ り ま す|。
火-星 猫|の|生-態|の|調-査 結-果
```

To train a model, use the following command:

```
% cargo run --release -p train -- --model ./your.model.zst --tok path/to/full.txt --part path/to/part.txt --dict path/to/dict.txt
```

`--tok` argument specifies a fully annotated corpus, and `--part` argument specifies a partially annotated corpus.
You can also specify a word dictionary with `--dict` argument.
A word dictionary is a file with words per line.

The trainer does not accept empty lines.
Therefore, remove all empty lines from the corpus before training.

You can specify all arguments above multiple times.

### Model Manipulation

Sometimes, your model will output different results than what you expect.
For example, `メロンパン` is split into two tokens in the following command.
We use `--scores` option to show the score of each character boundary:
```
% echo '朝食はメロンパン1個だった' | cargo run --release -p predict -- --scores --model path/to/jp-0.4.7-5-tokenize.model.zst
朝食 は メロン パン 1 個 だっ た
0:朝食 -15398
1:食は 24623
2:はメ 30261
3:メロ -26885
4:ロン -38896
5:ンパ 8162
6:パン -23416
7:ン１ 23513
8:１個 18435
9:個だ 24964
10:だっ -15065
11:った 14178
```

To concatenate `メロンパン` into a single token, manipulate the model in the following steps so that the score of `ンパ` becomes negative:

1. Dump a dictionary by the following command:
   ```
   % cargo run --release -p manipulate_model -- --model-in path/to/jp-0.4.7-5-tokenize.model.zst --dump-dict path/to/dictionary.csv
   ```

2. Edit the dictionary.

   The dictionary is a csv file. Each row contains a word, corresponding weights, and a comment in the following order:

   * `right_weight` - A weight that is added when the word is found to the right of the boundary.
   * `inside_weight` - A weight that is added when the word is overlapped on the boundary.
   * `left_weight` - A weight that is added when the word is found to the left of the boundary.
   * `comment` - A comment that does not affect the behaviour.

   Vaporetto splits a text when the total weight of the boundary is a positive number, so we add a new entry as follows:
   ```diff
    メロレオストーシス,6944,-2553,5319,
    メロン,8924,-10861,7081,
   +メロンパン,0,-100000,0,melon🍈 bread🍞 in English.
    メロン果実,4168,-1165,3558,
    メロヴィング,6999,-15413,7583,
   ```

   In this case, `-100000` will be added when the boundary is inside of the word `メロンパン`.

   Note that Vaporetto uses 32-bit integers for the total weight, so you have to be careful about overflow.

   In addition, The dictionary cannot contain duplicated words.
   When the word is already contained in the dictionary, you have to edit existing weights.

3. Replaces weight data of a model file
   ```
   % cargo run --release -p manipulate_model -- --model-in path/to/jp-0.4.7-5-tokenize.model.zst --replace-dict path/to/dictionary.csv --model-out path/to/jp-0.4.7-5-tokenize-new.model.zst
   ```

Now `メロンパン` is split into a single token.
```
% echo '朝食はメロンパン1個だった' | cargo run --release -p predict -- --scores --model path/to/jp-0.4.7-5-tokenize-new.model.zst
朝食 は メロンパン 1 個 だっ た
0:朝食 -15398
1:食は 24623
2:はメ 30261
3:メロ -126885
4:ロン -138896
5:ンパ -91838
6:パン -123416
7:ン１ 23513
8:１個 18435
9:個だ 24964
10:だっ -15065
11:った 14178
```

### POS tagging

Vaporetto experimentally supports POS tagging.

To train tags, add a slash and tag name following each token in the dataset as follows:

* For fully annotated corpora
  ```
  この/連体詞 人/名詞 は/助詞 火星/名詞 人/接尾辞 です/助動詞
  ```

* For partially annotated corpora
  ```
  ヴ-ェ-ネ-ツ-ィ-ア/名詞|は/助詞|イ-タ-リ-ア/名詞|に/助詞|あ-り ま-す
  ```

If the dataset contains tags, the `train` command automatically trains them.

In prediction, tags are not predicted by default, so you have to specify `--predict-tags` argument to `predict` command if necessary.

## Speed Comparison of Various Tokenizers

Vaporetto is 8.25 times faster than KyTea.

Details can be found [here](https://github.com/legalforce-research/vaporetto/wiki/Speed-Comparison).

![](./figures/comparison.svg)

## Disclaimer

This software is developed by LegalForce, Inc.,
but not an officially supported LegalForce product.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

## References

* Graham Neubig, Yosuke Nakata, and Shinsuke Mori. 2011. Pointwise prediction for
  robust, adaptable Japanese morphological analysis. In Proceedings of the 49th
  Annual Meeting of the Association for Computational Linguistics: Human Language
  Technologies: short papers - Volume 2 (HLT ‘11). Association for Computational
  Linguistics, USA, 529–533. https://aclanthology.org/P11-2093

* 森 信介, 中田 陽介, Neubig Graham, 河原 達也, 点予測による形態素解析, 自然言語処理, 2011, 18 巻,
  4 号, p. 367-381, 公開日 2011/12/28, Online ISSN 2185-8314, Print ISSN 1340-7619,
  https://doi.org/10.5715/jnlp.18.367

* Alfred V. Aho and Margaret J. Corasick. 1975. Efficient string matching: an aid to
  bibliographic search. Commun. ACM 18, 6 (June 1975), 333–340.
  DOI:https://doi.org/10.1145/360825.360855

* Jun-ichi Aoe. 1989. An Efficient Digital Search Algorithm by Using a Double-Array
  Structure. IEEE Trans. Softw. Eng. 15, 9 (September 1989), 1066–1077.
  DOI:https://doi.org/10.1109/32.31365
