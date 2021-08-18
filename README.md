# 🚤 VAporetto: POintwise pREdicTion based TOkenizer

## Usage

The following examples use [KFTT](http://www.phontron.com/kftt/) for training and prediction data.

### Training

Example:
```
% cargo run --release --bin train --  --model ./kftt.model --tok ./kftt-data-1.0/data/tok/kyoto-train.ja
```

### Prediction

Example:
```
% cargo run --release --bin predict -- --model ./kftt.model < ./kftt-data-1.0/data/orig/kyoto-test.ja > ./tokenized.ja
```

### Conversion from KyTea's Model File

Example:
```
% cargo run --release --bin convert_kytea_model -- --model-in ./jp-0.4.7-5.mod --model-out ./kytea.model
```
