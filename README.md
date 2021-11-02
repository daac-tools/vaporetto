# 🛥 VAporetto: POintwise pREdicTion based TOkenizer

Vaporetto is a fast and lightweight pointwise prediction based tokenizer.

[![Crates.io](https://img.shields.io/crates/v/vaporetto)](https://crates.io/crates/vaporetto)
[![Documentation](https://docs.rs/vaporetto/badge.svg)](https://docs.rs/vaporetto)
![Build Status](https://github.com/legalforce-research/vaporetto/actions/workflows/rust.yml/badge.svg)

[Technical details](https://tech.legalforce.co.jp/entry/2021/09/28/180844) (Japanese)

## Overview

This repository includes both a Rust crate that provides APIs for Vaporetto and CLI frontends.

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
% cargo run --release -p convert_kytea_model -- --model-in path/to/jp-0.4.7-5.mod --model-out path/to/jp-0.4.7-5-tokenize.model.zstd
```

Now you can perform tokenization. Run the following command:
```
% echo '火星猫の生態の調査結果' | cargo run --release -p predict -- --model path/to/jp-0.4.7-5-tokenize.model.zstd
```

The following will be output:
```
火星 猫 の 生態 の 調査 結果
```

#### Train Your Own Model

The second way, which is mainly for researchers, is to prepare your own training corpus and train your own tokenization models.

Vaporetto can train from two types of corpora: fully annotated corpora and partially annotated corpora.

Fully annotated corpora are corpora in which all character boundaries are annotated with token boundaries.
This is the data in the form of spaces inserted into the boundaries of the tokens, as shown below:

```
ヴェネツィア は イタリア に あ り ま す 。
火星 猫 の 生態 の 調査 結果
```

On the other hand,　partially annotated corpora are corpora in which only some character boundaries are annotated.
Each character boundary is annotated in the form of `|` (token boundary), `-` (not token boundary), and ` ` (unknown).
Here is an example:

```
ヴ-ェ-ネ-ツ-ィ-ア|は|イ-タ-リ-ア|に|あ り ま す|。
火-星 猫|の|生-態|の|調-査 結-果
```

To train a model, use the following command:

```
% cargo run --release -p train -- --model ./kftt.model.zstd --tok path/to/full.txt --part path/to/part.txt --dict path/to/dict.txt
```

`--tok` argument specifies a fully annotated corpus, and `--part` argument specifies a partially annotated corpus.
You can also specify a word dictionary with `--dict` argument.
A word dictionary is a file with words per line.

You can specify all arguments above multiple times.

## Speed Comparison of Various Tokenizers

### Experimental Setup

* Document: Japanese training data of Kyoto Free Translation Task
* Models:
  * KyTea and Vaporetto: Compact LR model (jp-0.4.7-6)
  * MeCab, Kuromoji, and Lindera: IPAdic
  * Sudachi and Sudachi.rs: system_core.dic (v20210802)

### Results

* VM instance on Google Cloud Platform (c2-standard-16, Debian)

  | Tool Name (version)        | Speed (×10^6 chars/s) | σ     |
  | -------------------------- | ---------------------:|-------|
  | KyTea (2020-04-03)         |                 0.777 | 0.020 |
  | Vaporetto (0.1.6)          |             **4.426** | 0.182 |
  |                            |                       |       |
  | MeCab (2020-09-14)         |                 2.736 | 0.041 |
  |                            |                       |       |
  | Kuromoji (Atilika's 0.9.0) |                 0.423 | 0.013 |
  | Lindera (0.8.0)            |                 1.002 | 0.014 |
  |                            |                       |       |
  | Sudachi (0.5.2)            |                 0.251 | 0.012 |
  | Sudachi.rs (0.6.0-rc1)     |                 0.644 | 0.012 |

* MacBook Pro (2017, Processor: 2.3 GHz Intel Core i5, Memory: 8 GB 2133 MHz LPDDR3)

  | Tool Name (version)        | Speed (×10^6 chars/s) | σ     |
  | -------------------------- | ---------------------:|-------|
  | KyTea (2020-04-03)         |                 0.490 | 0.003 |
  | Vaporetto (0.1.6)          |             **3.016** | 0.113 |
  |                            |                       |       |
  | MeCab (2020-09-14)         |                 1.418 | 0.007 |
  |                            |                       |       |
  | Kuromoji (Atilika's 0.9.0) |                 1.197 | 0.034 |
  | Lindera (0.8.0)            |                 0.542 | 0.010 |
  |                            |                       |       |
  | Sudachi (0.5.2)            |                 0.439 | 0.026 |
  | Sudachi.rs (0.6.0-rc1)     |                 0.427 | 0.009 |

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
