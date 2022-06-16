# 🛥 VAporetto: POintwise pREdicTion based TOkenizer

Vaporetto は、高速で軽量な点予測に基づくトークナイザです。
このリポジトリには、 Vaporetto の API を提供する Rust のクレートと、 CLI フロントエンドが含まれています。

[![Crates.io](https://img.shields.io/crates/v/vaporetto)](https://crates.io/crates/vaporetto)
[![Documentation](https://docs.rs/vaporetto/badge.svg)](https://docs.rs/vaporetto)
![Build Status](https://github.com/daac-tools/vaporetto/actions/workflows/rust.yml/badge.svg)

[English document](README.md)

[Wasm のデモ](https://daac-tools.github.io/vaporetto/) (モデルの読み込みに少し時間がかかります。)

## 使用例

### トークン化を試す

このソフトウェアは Rust で実装されています。事前に[ドキュメント](https://www.rust-lang.org/tools/install)に従って `rustc` と `cargo` をインストールしてください。

Vaporetto はトークン化モデルを生成するための方法を3つ用意しています。

#### 配布モデルをダウンロードする

1番目は最も単純な方法で、我々によって学習されたモデルをダウンロードすることです。
モデルファイルは[ここ](https://github.com/daac-tools/vaporetto/releases)にあります。

`bccwj-suw+unidic+tag` を選びました。
```
% wget https://github.com/daac-tools/vaporetto/releases/download/v0.5.0/bccwj-suw+unidic+tag.tar.xz
```

各ファイルにはモデルファイルとライセンス条項が含まれているので、以下のようなコマンドでダウンロードしたファイルを展開する必要があります。
```
% tar xf ./bccwj-suw+unidic+tag.tar.xz
```

トークン化を行うには、以下のコマンドを実行します。
```
% echo 'ヴェネツィアはイタリアにあります。' | cargo run --release -p predict -- --model path/to/bccwj-suw+unidic+tag.model.zst
```

以下が出力されるでしょう。
```
ヴェネツィア は イタリア に あり ます 。
```

#### KyTea のモデルを変換する

2番目の方法も単純で、 KyTea で学習されたモデルを変換することです。
まずはじめに、好きなモデルを [KyTea Models](http://www.phontron.com/kytea/model.html) ページからダウンロードします。

`jp-0.4.7-5.mod.gz` を選びました。
```
% wget http://www.phontron.com/kytea/download/model/jp-0.4.7-5.mod.gz
```

各モデルは圧縮されているので、以下のようなコマンドでダウンロードしたモデルを展開する必要があります。
```
% gunzip ./jp-0.4.7-5.mod.gz
```

KyTea のモデルを Vaporetto のモデルに変換するには、 Vaporetto のルートディレクトリで以下のコマンドを実行します。
```
% cargo run --release -p convert_kytea_model -- --model-in path/to/jp-0.4.7-5.mod --model-out path/to/jp-0.4.7-5-tokenize.model.zst
```

これでトークン化を行えます。以下のコマンドを実行します。
```
% echo 'ヴェネツィアはイタリアにあります。' | cargo run --release -p predict -- --model path/to/jp-0.4.7-5-tokenize.model.zst
```

以下が出力されるでしょう。
```
ヴェネツィア は イタリア に あ り ま す 。
```

#### 自分のモデルを学習する

3番目は主に研究者向けで、自分で学習コーパスを用意し、自分でトークン化モデルを学習することです。

Vaporetto は2種類のコーパス、すなわちフルアノテーションコーパスと部分アノテーションコーパスから学習することが可能です。

フルアノテーションコーパスは、すべての文字境界に対してトークン境界であるかトークンの内部であるかがアノテーションされたコーパスです。
このデータは、以下に示すようにトークン境界に空白が挿入された形式です。

```
ヴェネツィア は イタリア に あり ます 。
火星 猫 の 生態 の 調査 結果
```

一方、部分アノテーションコーパスは一部の文字境界のみに対してアノテーションされたコーパスです。
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
単語辞書は、1行1単語のファイルです。
辞書にはコーパスと同様にタグ情報を与えることができます。
モデルで推定できない単語には辞書で指定されたタグが付与されます。

学習器は空行の入力を受け付けません。
このため、学習の前にコーパスから空行を削除してください。

上記の引数は複数回指定することが可能です。

### モデルの編集

モデルが期待とは異なる結果を出力することがあるでしょう。
例えば、以下のコマンドで `外国人参政権` は誤ったトークンに分割されます。
`--scores` オプションを使って、各文字間のスコアを出力します。
```
% echo '外国人参政権と政権交代' | cargo run --release -p predict -- --scores --model path/to/bccwj-suw+unidic.model.zst
外国 人 参 政権 と 政権 交代
0:外国 -11785
1:国人 16634
2:人参 5450
3:参政 4480
4:政権 -3697
5:権と 17702
6:と政 18699
7:政権 -12742
8:権交 14578
9:交代 -7658
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
    参撾,3167 -6074 3790,
    参政,3167 -6074 3790,
   +参政権,0 -10000 10000 0,参政/権
    参朝,3167 -6074 3790,
    参校,3167 -6074 3790,
   ```

   この場合、 `参` と `政` の間に `-10000` が、 `政` と `権` の間に `10000` が加算されます。
   パターンの両端では `0` が指定されているため、スコアは加算されません。

   Vaporetto は重みの合計値に 32-bit 整数を利用しているため、オーバーフローに気をつけてください。

   加えて、辞書には重複する単語を含めることができません。
   単語が既に辞書に含まれている際は、既存の重みを編集する必要があります。

3. モデルファイルの重みを置換します。
   ```
   % cargo run --release -p manipulate_model -- --model-in path/to/bccwj-suw+unidic.model.zst --replace-dict path/to/dictionary.csv --model-out path/to/bccwj-suw+unidic-new.model.zst
   ```

これで `外国人参政権` が正しいトークンに分割されます。
```
% echo '外国人参政権と政権交代' | cargo run --release -p predict -- --scores --model path/to/bccwj-suw+unidic-new.model.zst
外国 人 参政 権 と 政権 交代
0:外国 -11785
1:国人 16634
2:人参 5450
3:参政 -5520
4:政権 6303
5:権と 17702
6:と政 18699
7:政権 -12742
8:権交 14578
9:交代 -7658
```

### 品詞推定

Vaporettoは実験的にタグ推定（品詞推定や読み推定）に対応しています。

タグを学習するには、以下のように、データセットの各トークンに続けてスラッシュとタグを追加します。

* フルアノテーションコーパスの場合
  ```
  この/連体詞/コノ 人/名詞/ヒト は/助詞/ワ 火星/名詞/カセイ 人/接尾辞/ジン です/助動詞/デス
  ```

* 部分アノテーションコーパスの場合
  ```
  ヴ-ェ-ネ-ツ-ィ-ア/名詞|は/助詞|イ-タ-リ-ア/名詞|に/助詞|あ-り ま-す
  ```

データセットにタグが含まれる場合、 `train` コマンドは自動的にそれらを学習します。

推定時は、デフォルトではタグは推定されないため、必要に応じで `predict` コマンドに `--predict-tags` 引数を指定してください。

## 各種トークナイザの速度比較

Vaporetto は KyTea に比べて 8.7 倍速く動作します。

詳細は[ここ](https://github.com/daac-tools/vaporetto/wiki/Speed-Comparison)を参照してください。

![](./figures/comparison.svg)

## 文献情報

Vaporettoにおける単語分割の仕組みについては、以下の論文またはブログ記事を参照してください。

 * 赤部晃一，神田峻介，小田悠介，森信介．
   [Vaporetto: 点予測法に基づく高速な日本語トークナイザ](https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/D2-5.pdf)．
   言語処理学会第28回年次大会(NLP2022)．浜松．2022年3月．
 * [速度の高みを目指す：高速な単語分割器 Vaporetto の技術解説](https://tech.legalforce.co.jp/entry/2021/09/28/180844) (技術ブログ)
