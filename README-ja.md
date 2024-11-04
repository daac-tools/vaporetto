# 🛥 Vaporetto: Very accelerated pointwise prediction based tokenizer

Vaporetto は、高速で軽量な点予測に基づくトークナイザです。
このリポジトリには、 Vaporetto の API を提供する Rust のクレートと、 CLI フロントエンドが含まれています。

[![Crates.io](https://img.shields.io/crates/v/vaporetto)](https://crates.io/crates/vaporetto)
[![Documentation](https://docs.rs/vaporetto/badge.svg)](https://docs.rs/vaporetto)
[![Build Status](https://github.com/daac-tools/vaporetto/actions/workflows/rust.yml/badge.svg)](https://github.com/daac-tools/vaporetto/actions)
[![Slack](https://img.shields.io/badge/join-chat-brightgreen?logo=slack)](https://join.slack.com/t/daac-tools/shared_invite/zt-1pwwqbcz4-KxL95Nam9VinpPlzUpEGyA)

[English document](README.md)

[Wasm のデモ](https://vaporetto-demo.pages.dev/) (モデルの読み込みに少し時間がかかります。)

## 使用例

### トークン化を試す

このソフトウェアは Rust で実装されています。事前に[ドキュメント](https://www.rust-lang.org/tools/install)に従って `rustc` と `cargo` をインストールしてください。

Vaporetto はトークン化モデルを生成するための方法を3つ用意しています。

#### 配布モデルをダウンロードする

1つ目は最も単純な方法で、学習済みモデルをダウンロードすることです。
モデルファイルは[ここ](https://github.com/daac-tools/vaporetto-models/releases)にあります。

`bccwj-suw+unidic+tag` を選びました。
```
% wget https://github.com/daac-tools/vaporetto-models/releases/download/v0.5.0/bccwj-suw+unidic_pos+pron.tar.xz
```

各ファイルはモデルファイルとライセンス条項が含まれた圧縮ファイルなので、ダウンロードしたファイルを展開する必要があります。
```
% tar xf ./bccwj-suw+unidic_pos+pron.tar.xz
```

トークン化には、以下のコマンドを実行します。
```
% echo 'ヴェネツィアはイタリアにあります。' | cargo run --release -p predict -- --model path/to/bccwj-suw+unidic_pos+pron.model.zst
```

以下が出力されます。
```
ヴェネツィア は イタリア に あり ます 。
```

##### Vaporetto APIs を使用する際の注意点

配布モデルは zstd 形式で圧縮されています。
*vaporetto* APIでこれらの圧縮済みモデルを読み込むには、APIの外側で展開する必要があります。

```rust
// zstd クレートまたは ruzstd クレートが必要
let reader = zstd::Decoder::new(File::open("path/to/model.zst")?)?;
let model = Model::read(reader)?;
```

最近のLinuxディストリビューションに同梱されている *unzstd* コマンドを利用して展開することもできます。

#### KyTea のモデルを変換する

2つ目の方法も単純で、 KyTea で学習されたモデルを変換することです。
まずはじめに、好きなモデルを [KyTea Models](http://www.phontron.com/kytea/model.html) ページからダウンロードします。

`jp-0.4.7-5.mod.gz` を選びました。
```
% wget http://www.phontron.com/kytea/download/model/jp-0.4.7-5.mod.gz
```

各モデルは圧縮されているので、ダウンロードしたモデルを展開する必要があります。
```
% gunzip ./jp-0.4.7-5.mod.gz
```

KyTea のモデルを Vaporetto のモデルに変換するには、 Vaporetto のルートディレクトリで以下のコマンドを実行します。
```
% cargo run --release -p convert_kytea_model -- --model-in path/to/jp-0.4.7-5.mod --model-out path/to/jp-0.4.7-5-tokenize.model.zst
```

これでトークン化できます。以下のコマンドを実行します。
```
% echo 'ヴェネツィアはイタリアにあります。' | cargo run --release -p predict -- --model path/to/jp-0.4.7-5-tokenize.model.zst
```

以下が出力されます。
```
ヴェネツィア は イタリア に あ り ま す 。
```

#### 自分のモデルを学習する

3つ目は主に研究者向けで、自分で学習コーパスを用意し、モデルを学習することです。

Vaporetto は2種類のコーパス（フルアノテーションコーパスと部分アノテーションコーパス）から学習することが可能です。

フルアノテーションコーパスは、すべての文字境界に対してトークン境界であるかトークンの内部であるかがアノテーションされたコーパスです。
このデータは、以下に示すようにトークン境界に空白が挿入された形式です。

```
ヴェネツィア は イタリア に あり ます 。
火星 猫 の 生態 の 調査 結果
```

部分アノテーションコーパスは、一部の文字境界のみに対してアノテーションされたコーパスです。
各文字境界には `|` (トークン境界)、 `-` (非トークン境界)、 ` ` (不明) のいずれかの形式でアノテーションされます。

ここに例を示します。
```
ヴ-ェ-ネ-ツ-ィ-ア|は|イ-タ-リ-ア|に|あ り ま す|。
火-星 猫|の|生-態|の|調-査 結-果
```

モデルを学習するには、以下のコマンドを使用します。
```
% cargo run --release -p train -- --model ./your.model.zst --tok path/to/full.txt --part path/to/part.txt --dict path/to/dict.txt --solver 5
```

`--tok` 引数ではフルアノテーションコーパスを指定し、 `--part` 引数では部分アノテーションコーパスを指定します。
`--dict` 引数によって単語辞書を指定することもできます。
単語辞書は、1行1単語のファイルであり、必要に応じてタグを付与することもできます。
```
トスカーナ
パンツァーノ
灯里/名詞-固有名詞-人名-名/アカリ
形態/名詞-普通名詞-一般/ケータイ
```

学習器は空行の入力を受け付けません。
このため、学習の前にコーパスから空行を削除してください。

上記の引数は複数回指定することが可能です。

### モデルの編集

モデルが期待とは異なる結果を出力することがあるでしょう。
例えば、以下のコマンドで `外国人参政権` は誤ったトークンに分割されます。
`--scores` オプションを使って、各文字間のスコアを出力します。
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

正しくは `外国 人 参政 権` です。
`外国人参政権` を正しいトークンに分割するには、以下の手順でモデルを編集し、 `参政権` のスコアの符号を反転させます。

1. 以下のコマンドで辞書を吐き出します。
   ```
   % cargo run --release -p manipulate_model -- --model-in path/to/bccwj-suw+unidic.model.zst --dump-dict path/to/dictionary.csv
   ```

2. 辞書を編集します。

   辞書は CSV ファイルです。各行には文字列パターン、対応する重み配列、コメントが以下のように含まれています。

   * `word` - 文字列パターン（主に単語）
   * `weights` - 重み配列。入力文字列に対象の文字列パターンが含まれている場合、見つかったパターンの範囲の文字境界に対してこれらの重みが加算されます。
   * `comment` - 挙動に影響しないコメント

   Vaporetto は、重みの合計が正の値になった際にテキストを分割するので、以下のように新しいエントリを追加します。
   ```diff
    参撾,3328 -5545 3514,
    参政,3328 -5545 3514,
   +参政権,0 -10000 10000 0,参政/権
    参朝,3328 -5545 3514,
    参校,3328 -5545 3514,
   ```

   この場合、 `参` と `政` の間に `-10000` が、 `政` と `権` の間に `10000` が加算されます。
   パターンの両端では `0` が指定されているため、スコアは加算されません。

   Vaporetto は重みの合計値に 32-bit 整数を利用しているため、オーバーフローに気をつけてください。

   加えて、辞書には重複する単語を含めることができません。
   単語が既に辞書に含まれている際は、既存の重みを編集する必要があります。

3. モデルファイルの重みを置換します。
   ```
   % cargo run --release -p manipulate_model -- --model-in path/to/bccwj-suw+unidic_pos+pron.model.zst --replace-dict path/to/dictionary.csv --model-out path/to/bccwj-suw+unidic_pos+pron-new.model.zst
   ```

これで `外国人参政権` が正しいトークンに分割されます。
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

### タグ予測

Vaporettoは実験的にタグ予測（品詞予測や読み予測）に対応しています。

タグを学習するには、以下のように、データセットの各トークンに続けてスラッシュとタグを追加します。

* フルアノテーションコーパスの場合
  ```
  この/連体詞/コノ 人/名詞/ヒト は/助詞/ワ 火星/名詞/カセイ 人/接尾辞/ジン です/助動詞/デス
  ```

* 部分アノテーションコーパスの場合
  ```
  ヴ-ェ-ネ-ツ-ィ-ア/名詞|は/助詞|イ-タ-リ-ア/名詞|に/助詞|あ-り ま-す
  ```

タグ情報は辞書に対してもコーパスと同様に与えることができます。
モデルで推定できない単語には辞書で指定されたタグが付与されます。

データセットにタグが含まれる場合、 `train` コマンドは自動的にそれらを学習します。

予測時は、デフォルトではタグは予測されないため、必要に応じて `predict` コマンドに `--predict-tags` 引数を指定してください。

`--tag-scores` 引数を指定すると、タグ予測の際に計算された各候補のスコアを表示できます。
タグの候補が1つしかない場合は、スコアが0と表示されます。

```
% echo "花が咲く" | cargo run --release -p predict -- --model path/to/bccwj-suw+unidic_pos+pron.model.zst --predict-tags --tag-scores
花/名詞-普通名詞-一般/ハナ が/助詞-格助詞/ガ 咲く/動詞-一般/サク
花	名詞-普通名詞-一般:18613,接尾辞-名詞的-一般:-18613	ハナ:19973,バナ:-20377,カ:-20480,ゲ:-20410
が	助詞-接続助詞:-20408,助詞-格助詞:23543,接続詞:-25332	ガ:0
咲く	動詞-一般:0	サク:0
```

## 各種トークナイザの速度比較

Vaporetto は KyTea に比べて 8.7 倍速く動作します。

詳細は[ここ](https://github.com/daac-tools/vaporetto/wiki/Speed-Comparison)を参照してください。

![](./figures/comparison.svg)

## Slack

開発者やユーザーの方々が質問したり議論するためのSlackワークスペースを用意しています。

 * https://daac-tools.slack.com/
 * [こちら](https://join.slack.com/t/daac-tools/shared_invite/zt-1pwwqbcz4-KxL95Nam9VinpPlzUpEGyA)から招待を受けてください.

## 文献情報

Vaporettoにおける単語分割の仕組みについては、以下の論文またはブログ記事を参照してください。

 * Koichi Akabe, Shunsuke Kanda, Yusuke Oda, Shinsuke Mori.
   [Vaporetto: Efficient Japanese Tokenization Based on Improved Pointwise Linear Classification](https://arxiv.org/abs/2406.17185).
   arXiv. 2024.
 * 赤部晃一，神田峻介，小田悠介，森信介．
   [Vaporetto: 点予測法に基づく高速な日本語トークナイザ](https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/D2-5.pdf)．
   言語処理学会第28回年次大会(NLP2022)．浜松．2022年3月．
 * [速度の高みを目指す：高速な単語分割器 Vaporetto の技術解説](https://tech.legalforce.co.jp/entry/2021/09/28/180844) (技術ブログ)

Vaporettoで使用しているダブル配列Aho-Corasick法 (DAAC) の技術情報については、以下の論文を参照してください。

 * Shunsuke Kanda, Koichi Akabe, and Yusuke Oda.
   [Engineering faster double-array Aho-Corasick automata](https://doi.org/10.1002/spe.3190). Software: Practice and Experience (SPE),
   53(6): 1332–1361, 2023 ([arXiv](https://arxiv.org/abs/2207.13870))
