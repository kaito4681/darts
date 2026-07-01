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

### 使える主な引数

全引数は各スクリプトの `--help` でも確認できます。

```bash
uv run cnn/train_search.py --help
uv run rnn/train.py --help
```

#### 共通

| 引数                | 対象                 | 内容                        |
| ------------------- | -------------------- | --------------------------- |
| `--data PATH`       | 全実験               | データセットのパス          |
| `--epochs N`        | train / train_search | エポック数                  |
| `--batch_size N`    | 全実験               | バッチサイズ                |
| `--gpu N`           | 全実験               | 使用する GPU ID             |
| `--seed N`          | 全実験               | 乱数 seed                   |
| `--save NAME`       | train / train_search | 出力ディレクトリ名の prefix |
| `--model_path PATH` | test                 | 事前学習モデルのパス        |

#### CNN

| 引数                     | 対象                    | 内容                                   |
| ------------------------ | ----------------------- | -------------------------------------- |
| `--unrolled`             | `cnn/train_search.py`   | second-order / unrolled の探索を使う   |
| `--train_portion R`      | `cnn/train_search.py`   | CIFAR-10 train split の探索用比率      |
| `--learning_rate R`      | train / train_search    | 学習率                                 |
| `--learning_rate_min R`  | `cnn/train_search.py`   | cosine scheduler の最小学習率          |
| `--momentum R`           | train / train_search    | SGD momentum                           |
| `--weight_decay R`       | train / train_search    | weight decay                           |
| `--report_freq N`        | 全 CNN 実験             | ログ出力間隔                           |
| `--arch_learning_rate R` | `cnn/train_search.py`   | architecture parameter の学習率        |
| `--arch_weight_decay R`  | `cnn/train_search.py`   | architecture parameter の weight decay |
| `--init_channels N`      | 全 CNN 実験             | 初期 channel 数                        |
| `--layers N`             | 全 CNN 実験             | layer 数                               |
| `--arch NAME`            | train / test            | `cnn/genotypes.py` の architecture 名  |
| `--auxiliary`            | train / test            | auxiliary tower を使う                 |
| `--auxiliary_weight R`   | train                   | auxiliary loss の重み                  |
| `--cutout`               | CIFAR train / test      | cutout augmentation を使う             |
| `--cutout_length N`      | CIFAR train / test      | cutout の長さ                          |
| `--drop_path_prob R`     | train / test            | drop path 確率                         |
| `--grad_clip R`          | train / train_search    | gradient clipping の閾値               |
| `--parallel`             | `cnn/train_imagenet.py` | DataParallel を使う                    |
| `--label_smooth R`       | `cnn/train_imagenet.py` | label smoothing                        |
| `--gamma R`              | `cnn/train_imagenet.py` | StepLR の decay 率                     |
| `--decay_period N`       | `cnn/train_imagenet.py` | learning rate decay の周期             |

#### RNN

| 引数                    | 対象                  | 内容                                     |
| ----------------------- | --------------------- | ---------------------------------------- |
| `--unrolled`            | `rnn/train_search.py` | second-order / unrolled の探索を使う     |
| `--emsize N`            | 全 RNN 実験           | embedding size                           |
| `--nhid N`              | 全 RNN 実験           | hidden size                              |
| `--nhidlast N`          | train / train_search  | 最終 RNN layer の hidden size            |
| `--lr R`                | train / train_search  | 学習率                                   |
| `--clip R`              | train / train_search  | gradient clipping の閾値                 |
| `--bptt N`              | 全 RNN 実験           | BPTT の sequence length                  |
| `--dropout R`           | 全 RNN 実験           | layer dropout                            |
| `--dropouth R`          | 全 RNN 実験           | hidden state dropout                     |
| `--dropoutx R`          | train / train_search  | RNN input dropout                        |
| `--dropouti R`          | 全 RNN 実験           | embedding input dropout                  |
| `--dropoute R`          | 全 RNN 実験           | embedding dropout                        |
| `--alpha R`             | 全 RNN 実験           | activation regularization                |
| `--beta R`              | 全 RNN 実験           | temporal activation regularization       |
| `--wdecay R`            | train / train_search  | weight decay                             |
| `--log-interval N`      | 全 RNN 実験           | ログ出力間隔                             |
| `--small_batch_size N`  | train / train_search  | gradient accumulation 用の小バッチサイズ |
| `--max_seq_len_delta N` | 全 RNN 実験           | sequence length の揺らぎ幅               |
| `--single_gpu`          | train / train_search  | 指定すると DataParallel 実行に切り替える |
| `--cuda`                | 全 RNN 実験           | 指定すると CUDA を使わない                |
| `--continue_train`      | train / train_search  | checkpoint から再開                      |
| `--arch NAME`           | `rnn/train.py`        | `rnn/genotypes.py` の architecture 名    |
| `--arch_lr R`           | `rnn/train_search.py` | architecture parameter の学習率          |
| `--arch_wdecay R`       | `rnn/train_search.py` | architecture parameter の weight decay   |

### 可視化

```bash
uv run cnn/visualize.py DARTS
uv run rnn/visualize.py DARTS
```

`DARTS` は `genotypes.py` に定義した任意のアーキテクチャ名に置き換えられます。
