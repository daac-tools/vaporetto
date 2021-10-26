# 🛥 VAporetto: POintwise pREdicTion based TOkenizer

Vaporetto is a fast and lightweight pointwise prediction based tokenizer.

[![Crates.io](https://img.shields.io/crates/v/vaporetto)](https://crates.io/crates/vaporetto)
[![Documentation](https://docs.rs/vaporetto/badge.svg)](https://docs.rs/vaporetto)

[Technical details](https://tech.legalforce.co.jp/entry/2021/09/28/180844) (Japanese)

## Overview

This repository includes both a Rust crate that provides APIs for Vaporetto and CLI frontends.

The following examples use [KFTT](http://www.phontron.com/kftt/) for training and prediction data.

### Training

```
% cargo run --release --bin train -- --model ./kftt.model.zstd --tok ./kftt-data-1.0/data/tok/kyoto-train.ja
```

### Prediction

```
% cargo run --release --bin predict -- --model ./kftt.model.zstd < ./kftt-data-1.0/data/orig/kyoto-test.ja > ./tokenized.ja
```

### Conversion from KyTea's Model File

```
% cargo run --release --bin convert_kytea_model -- --model-in ./jp-0.4.7-5.mod --model-out ./kytea.model.zstd
```

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
  | KyTea (0.4.7)              |                 0.777 | 0.020 |
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
  | KyTea (0.4.7)              |                 0.500 | 0.008 |
  | Vaporetto (0.1.5)          |             **2.773** | 0.103 |
  |                            |                       |       |
  | MeCab (2020-09-14)         |                 1.413 | 0.018 |
  |                            |                       |       |
  | Kuromoji (Atilika's 0.9.0) |                 1.219 | 0.013 |
  | Lindera (0.8.0)            |                 0.547 | 0.014 |
  |                            |                       |       |
  | Sudachi (0.5.2)            |                 0.445 | 0.026 |
  | Sudachi.rs (2021-10-01)    |                 0.147 | 0.002 |

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
