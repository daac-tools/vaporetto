# 🛥 VAporetto: POintwise pREdicTion based TOkenizer

Vaporetto は、高速で軽量な点予測に基づくトークナイザです。
このリポジトリには、 Vaporetto の API を提供する Rust のクレートと、 CLI フロントエンドが含まれています。

[![Crates.io](https://img.shields.io/crates/v/vaporetto)](https://crates.io/crates/vaporetto)
[![Documentation](https://docs.rs/vaporetto/badge.svg)](https://docs.rs/vaporetto)
![Build Status](https://github.com/daac-tools/vaporetto/actions/workflows/rust.yml/badge.svg)

[技術解説](https://tech.legalforce.co.jp/entry/2021/09/28/180844)

[English document](README.md)

## 使用例

### トークン化を試す

このソフトウェアは Rust で実装されています。事前に[ドキュメント](https://www.rust-lang.org/tools/install)に従って `rustc` と `cargo` をインストールしてください。

Vaporetto はトークン化モデルを生成するための方法を3つ用意しています。

#### 配布モデルをダウンロードする

1番目は最も単純な方法で、我々によって学習されたモデルをダウンロードすることです。
モデルファイルは[ここ](https://github.com/daac-tools/vaporetto/releases/tag/v0.3.0)にあります。

`bccwj-suw+unidic+tag` を選びました。
```
% wget https://github.com/daac-tools/vaporetto/releases/download/v0.3.0/bccwj-suw+unidic+tag.tar.xz
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
% cargo run --release -p train -- --model ./your.model.zst --tok path/to/full.txt --part path/to/part.txt --dict path/to/dict.txt
```

`--tok` 引数ではフルアノテーションコーパスを指定し、 `--part` 引数では部分アノテーションコーパスを指定します。
`--dict` 引数によって単語辞書を指定することもできます。
単語辞書は、1行1単語のファイルです。

学習器は空行の入力を受け付けません。
このため、学習の前にコーパスから空行を削除してください。

上記の引数は複数回指定することが可能です。

### モデルの編集

時々、モデルが期待とは異なる結果を出力することがあるでしょう。
例えば、以下のコマンドで `メロンパン` は2つのトークンに分割されます。
`--scores` オプションを使って、各文字間のスコアを出力します。
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

`メロンパン` を単一のトークンに連結するには、以下の手順でモデルを編集し、 `ンパ` のスコアを負にします。

1. 以下のコマンドで辞書を吐き出します。
   ```
   % cargo run --release -p manipulate_model -- --model-in path/to/jp-0.4.7-5-tokenize.model.zst --dump-dict path/to/dictionary.csv
   ```

2. 辞書を編集します。

   辞書は CSV ファイルです。各行には単語と、対応する重みとコメントが以下の順で含まれています。

   * `right_weight` - 単語が境界の右側に見つかった際に追加される重み。
   * `inside_weight` - 単語が境界に重なっている際に追加される重み。
   * `left_weight` - 単語が境界の左側に見つかった際に追加される重み。
   * `comment` - 挙動に影響しないコメント

   Vaporetto は、重みの合計が正の値になった際にテキストを分割するので、以下のように新しいエントリを追加します。
   ```diff
    メロレオストーシス,6944,-2553,5319,
    メロン,8924,-10861,7081,
   +メロンパン,0,-100000,0,melon🍈 bread🍞 in English.
    メロン果実,4168,-1165,3558,
    メロヴィング,6999,-15413,7583,
   ```

   この場合、境界が `メロンパン` の内側だった際に `-100000` が追加されます。

   Vaporetto は重みの合計値に 32-bit 整数を利用しているため、オーバーフローに気をつけてください。

   加えて、辞書には重複する単語を含めることができません。
   単語が既に辞書に含まれている際は、既存の重みを編集する必要があります。

3. モデルファイルの重みを置換します。
   ```
   % cargo run --release -p manipulate_model -- --model-in path/to/jp-0.4.7-5-tokenize.model.zst --replace-dict path/to/dictionary.csv --model-out path/to/jp-0.4.7-5-tokenize-new.model.zst
   ```

これで `メロンパン` が単一のトークンに分割されます。
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

### 品詞推定

Vaporettoは実験的に品詞推定に対応しています。

品詞を学習するには、以下のように、データセットの各トークンに続けてスラッシュと品詞を追加します。

* フルアノテーションコーパスの場合
  ```
  この/連体詞 人/名詞 は/助詞 火星/名詞 人/接尾辞 です/助動詞
  ```

* 部分アノテーションコーパスの場合
  ```
  ヴ-ェ-ネ-ツ-ィ-ア/名詞|は/助詞|イ-タ-リ-ア/名詞|に/助詞|あ-り ま-す
  ```

データセットに品詞が含まれる場合、 `train` コマンドは自動的にそれらを学習します。

推定時は、デフォルトでは品詞は推定されないため、必要に応じで `predict` コマンドに `--predict-tags` 引数を指定してください。

## 各種トークナイザの速度比較

Vaporetto は KyTea に比べて 8.25 倍速く動作します。

詳細は[ここ](https://github.com/daac-tools/vaporetto/wiki/Speed-Comparison)を参照してください。

![](./figures/comparison.svg)
