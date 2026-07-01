# DARTS: Differentiable Architecture Search

[DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) の実装です。

<p align="center">
  <img src="img/darts.png" alt="darts" width="48%">
</p>

## セットアップ

```bash
uv sync
```

Linux / Windows では `torch` と `torchvision` を PyTorch 公式の CUDA 12.8 wheel index からインストールします。

エラーが出たら，

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install graphviz
```

## 実行形式

各スクリプトは `cnn/` または `rnn/` をカレントディレクトリにして実行する前提です。リポジトリルートから `uv` で実行する場合は、次の形式を使います。

```bash
uv run cnn/<script>.py [args]
uv run rnn/<script>.py [args]
```

## データセット

- CIFAR-10: `torchvision` から自動ダウンロードされます。
- PTB / WikiText-2: [AWD-LSTM-LM](https://github.com/salesforce/awd-lstm-lm) の手順を参考に配置してください。
- ImageNet: [pytorch/examples の ImageNet 手順](https://github.com/pytorch/examples/tree/master/imagenet) を参考に配置してください。（長谷川研のNASにあります）

デフォルトのデータパス:

- CNN: `data/`
- RNN PTB: `data/penn/`
- RNN WikiText-2: `data/wikitext-2/`
- ImageNet: `data/imagenet/`

## 実験コマンド

### アーキテクチャ探索

```bash
# CIFAR-10
uv run cnn/train_search.py --data data --unrolled

# PTB
uv run rnn/train_search.py --data data/penn --unrolled
```

### アーキテクチャ評価

```bash
# CIFAR-10
uv run cnn/train.py --data data --auxiliary --cutout

# PTB
uv run rnn/train.py --data data/penn

# WikiText-2
uv run rnn/train.py --data data/wikitext-2 --dropouth 0.15 --emsize 700 --nhidlast 700 --nhid 700 --wdecay 5e-7

# ImageNet
uv run cnn/train_imagenet.py --data data/imagenet --auxiliary
```

### 事前学習モデルの評価

```bash
# CIFAR-10
uv run cnn/test.py --data data --auxiliary --model_path cifar10_model.pt

# PTB
uv run rnn/test.py --data data/penn --model_path ptb_model.pt

# ImageNet
uv run cnn/test_imagenet.py --data data/imagenet --auxiliary --model_path imagenet_model.pt
```

### 可視化

```bash
uv run cnn/visualize.py DARTS
uv run rnn/visualize.py DARTS
```

`DARTS` は `genotypes.py` に定義した任意のアーキテクチャ名に置き換えられます。
