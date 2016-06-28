---
layout: post
title:  "3. 手書き文字認識モデルの学習"
date:   2016-07-16 03:00:00 +09:00
category: caffe
permalink: 3-lenet
---

* TOC
{:toc}

このセクションではCaffeとPythonの`Solver` APIを使って学習を行ってみましょう。今回はMNISTという、標準的な手書き文字の認識用ベンチマークデータを用います。このデータセットからLeNet (LeCun 1989)アーキテクチャのニューラルネットワークを学習してみます。

このセクションの内容はGPUを使わない以外は`caffe/examples/01-learning-lenet.ipynb`と同等です。

## 準備

まずは`pylab`をインポートして`numpy`と`matplotlib`を同時に使えるようにします。

```python
from pylab import *
%matplotlib inline
```

続いて`caffe`をインポートします。

```python
import caffe

caffe_root = '/home/user/caffe/'
```

今回はMNISTデータセットを使います。以下の作業をしてLMDB形式でMNISTデータセットを使えるようにします。

```python
# スクリプトをCaffeのルートディレクトリから動かします
import os
os.chdir(caffe_root)
# MNISTデータセットをダウンロードします（この実習ではすでにダウンロード済みです）
!data/mnist/get_mnist.sh
# MNISTデータセットをCaffeで使えるようにLMDBフォーマットに変換します
!examples/mnist/create_mnist.sh
# 元の場所に戻ってきましょう
os.chdir('examples')
```

## ニューラルネットワークを構成

それでは古典的な畳み込みネットワークである[LeNet](http://yann.lecun.com/exdb/lenet/)をCaffeで作ってみましょう。

ネットワークを学習するには二つのファイルが必要になります。

 * 学習またはテスト時のネットワークの構造を記述した`prototxt`ファイル
 * 学習アルゴリズムのパラメータを指定した`prototxt`ファイル

最初にネットワーク構造を作るところから始めます。ここではPythonを使ってネットワークを記述し、protobufフォーマットに変換する方法でやります。学習時はネットワークの入力としてLMDBデータベースに学習データが保存されていることを想定していますが、これは`MemoryDataLayer`を使ってメモリから直接`ndarray`を読み出すように設定することもできます。

```python
from caffe import layers as L, params as P

def lenet(lmdb, batch_size):
    # Caffe版LeNet: 線形変換と非線形変換を組み合わせたもの
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()

with open('mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_train_lmdb', 64)))

with open('mnist/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_test_lmdb', 100)))
```

最初にLeNetの構造を作る関数`lenet`を定義し、最後の二つの箇所で`lenet`関数で作ったネットワークの記述を`prototxt`ファイルに書き出しています。ネットワーク構造を作るところはCaffeの`layers`と`params`クラスを用いています。書き出したファイルはGoogleのProtocol bufferというフォーマットを人が読んでも理解できるようなテキスト形式で保存しています。`prototxt`ファイルは直接テキストエディタで開いて編集することも可能です。学習用のネットワーク構造を見てみましょう。

```python
!cat mnist/lenet_auto_train.prototxt
```

```
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  transform_param {
    scale: 0.00392156862745
  }
  data_param {
    source: "mnist/mnist_train_lmdb"
    batch_size: 64
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 20
    kernel_size: 5
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  convolution_param {
    num_output: 50
    kernel_size: 5
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "fc1"
  type: "InnerProduct"
  bottom: "pool2"
  top: "fc1"
  inner_product_param {
    num_output: 500
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "fc1"
  top: "fc1"
}
layer {
  name: "score"
  type: "InnerProduct"
  bottom: "fc1"
  top: "score"
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "score"
  bottom: "label"
  top: "loss"
}
```

Prototxtの中身にはレイヤー単位でレイヤーの名前やタイプ、入出力、パラメータなどのネットワーク構造の詳細が記述されていることがわかると思います。レイヤーの記述にある`top`はそのレイヤーの出力、`bottom`は入力の名前で、同じ名前のものがあるレイヤーが入出力接続されます。

学習パラメータの`prototxt`ファイルも見てみましょう。今回はCaffeにあらかじめ用意されているファイルを使います。ここでは確率的勾配降下法（SGD）を使うためのモーメンタム(`momentum`)、重み減衰(`weight decay`)、それから学習率のスケジュール(`lr_policy`)などを指定します。

```python
!cat mnist/lenet_auto_solver.prototxt
```

```
# The train/test net protocol buffer definition
train_net: "mnist/lenet_auto_train.prototxt"
test_net: "mnist/lenet_auto_test.prototxt"
# test_iter specifies how many forward passes the test should carry out.
# In the case of MNIST, we have test batch size 100 and 100 test iterations,
# covering the full 10,000 testing images.
test_iter: 100
# Carry out testing every 500 training iterations.
test_interval: 500
# The base learning rate, momentum and the weight decay of the network.
base_lr: 0.01
momentum: 0.9
weight_decay: 0.0005
# The learning rate policy
lr_policy: "inv"
gamma: 0.0001
power: 0.75
# Display every 100 iterations
display: 100
# The maximum number of iterations
max_iter: 10000
# snapshot intermediate results
snapshot: 5000
snapshot_prefix: "mnist/lenet"
```

## ソルバーの読み込みと確認

それではソルバーを読み込みましょう。ここではモーメンタム付きのSGEアルゴリズムを用いますが、他の学習アルゴリズム（AdagradやNesterov's accelerated gradientなど）も使うことができます。

```python
### ソルバーを読み込んで学習とテスト用のネットワークを作成
solver = None  # LMDBデータを扱うときの制限回避のために変数を初期化します
solver = caffe.SGDSolver('mnist/lenet_auto_solver.prototxt')
```

作成したネットワークの中身は画像分類のセクションと同様に`blobs`と`params`から確認することができます。

```python
# 出力のブロブ形状は (batch size, feature dim, spatial dim)
[(k, v.data.shape) for k, v in solver.net.blobs.items()]
```

```
[('data', (64, 1, 28, 28)),
 ('label', (64,)),
 ('conv1', (64, 20, 24, 24)),
 ('pool1', (64, 20, 12, 12)),
 ('conv2', (64, 50, 8, 8)),
 ('pool2', (64, 50, 4, 4)),
 ('fc1', (64, 500)),
 ('score', (64, 10)),
 ('loss', ())]
```

```python
# 重みの大きさを表示
[(k, v[0].data.shape) for k, v in solver.net.params.items()]
```

```
[('conv1', (20, 1, 5, 5)),
 ('conv2', (50, 20, 5, 5)),
 ('fc1', (500, 800)),
 ('score', (10, 500))]
```

本格的に学習を始める前に、全てちゃんと動作しているか確かめてみましょう。一度だけ学習用とテスト用のネットワークでネットワークにフォワードパスでデータを流し、ちゃんとデータが読み込まれているか見てみます。

```python
solver.net.forward()  # 学習用ネットワークに流す
solver.test_nets[0].forward()  # テスト用ネットワークに流す（複数回やって構いません）
```

画像が正しく読み込まれているか見てみます。ブロブに含まれるデータから画像のピクセル配列をうまく抜き出して整形表示していることに注意してください。

```python
# 学習用ネットワークから最初の8枚の画像を表示
imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
print 'train labels:', solver.net.blobs['label'].data[:8]
```

`train labels: [ 5.  0.  4.  1.  9.  2.  1.  3.]`

![mnist-train-images]({{ site.baseurl }}/assets/3-mnist-train-images.png)

```python
imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
print 'test labels:', solver.test_nets[0].blobs['label'].data[:8]
```

`test labels: [ 7.  2.  1.  0.  4.  1.  4.  9.]`

![mnist-test-images]({{ site.baseurl }}/assets/3-mnist-test-images.png)

学習用、テスト用のネットワークが正しくデータを読み込み、正しいラベルを受け取っていることが確認できたました。

## ソルバーのステップ

それではSGDの（ミニバッチの）1ステップを動かしてみましょう。バッチサイズ50なら50枚の画像を読み込んでバックプロパゲーションを行います。

```python
solver.step(1)
```

勾配はネットワークを伝わっていったでしょうか？最初のレイヤーの重み更新を5x5の大きさのフィルタ群を4x5のグリッド状に並べて可視化してみましょう。

```python
imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5)
       .transpose(0, 2, 1, 3).reshape(4*5, 5*5), cmap='gray'); axis('off')
```

![lenet-gradients]({{ site.baseurl }}/assets/3-lenet-gradients.png)

どうやら重みの更新には非ゼロの値が伝わっていっているようですね。

## 学習ループを構成する

しばらくの間重みの更新を続け、その様子を観察してみましょう。PythonのソルバーはC++の`caffe`バイナリプログラムと同様の動作します。

  * 学習のステップが進むごとにログ出力
  * 現在のモデルのスナップショットをsolve prototxtに指定されている一定間隔で書き出し（今回は5000イテレーション毎）
  * 現在のモデルのテストデータでの性能評価は指定された間隔で行う（今回は500イテレーション毎）

Pythonでループの制御ができるので、追加で好きなように処理を行うことができます。例えば特別な反復停止条件を付け加えたり、ネットワークに変更を加えて学習のプロセスを変えてみる、などです。以下に少しテスト処理を付け加えたループの例を載せます。

```python
%%time
niter = 200
test_interval = 25
# 損失関数(loss)の値はログに出力されます
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

# メインのソルバーループです
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # 学習ロスを記録
    train_loss[it] = solver.net.blobs['loss'].data

    # 最初のテストバッチの出力を記録
    # （新しいデータを読み込まないようにフォワード処理をconv1でやる）
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['score'].data[:8]

    # テストを独自の指定間隔でやる
    # （Caffeでは自動でこれをやってくれますが、今回はループの書き方の一例としてあえて
    # 　手動でPythonからテストをやっています）
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4
```

学習データとテストデータの損失関数の変化をプロットしてみましょう。

```python
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
```

![lenet-loss1]({{ site.baseurl }}/assets/3-lenet-loss1.png)

ロスは急速に減少して収束し（確率的な変動は除いて）、正答率はこれに対応して上昇しているようですね。よしよし。

ここでは最初のテストバッチの出力を保存してきたので、予測のスコアが学習を経てどのように変化したのかを見ることができます。x軸に時間方向の変化を、y軸にラベル毎の推定スコアの変化を明るさでプロットしてみましょう。

```python
for i in range(8):
    figure(figsize=(2, 2))
    imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    figure(figsize=(10, 2))
    imshow(output[:50, i].T, interpolation='nearest', cmap='gray')
    xlabel('iteration')
    ylabel('label')
```

![mnist-7]({{ site.baseurl }}/assets/3-mnist-7.png)

![mnist-7-evolution]({{ site.baseurl }}/assets/3-mnist-7-evolution.png)

最初は数字について何も知らない状態から始めましたが、各桁について正しい識別ができるようになりました。出力結果をよく見ている人は、最後の傾いた_9_が_4_と混乱しやすいことに気づいたかもしれません。

上の結果はSoftmax後の確率ではなく生の出力なので少し見にくいかもしれません。以下のようにするともう少しはっきりとスコアを見ることができるかもしれません。（ただし小さなスコアの見分けがつきにくくなります）

```python
for i in range(8):
    figure(figsize=(2, 2))
    imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    figure(figsize=(10, 2))
    imshow(exp(output[:50, i].T) / exp(output[:50, i].T).sum(0), interpolation='nearest', cmap='gray')
    xlabel('iteration')
    ylabel('label')
```

![mnist-7]({{ site.baseurl }}/assets/3-mnist-7.png)

![mnist-7-evolution]({{ site.baseurl }}/assets/3-mnist-7-evolution-softmax.png)

## 様々なネットワーク構造と最適化を試行

ここまでLeNetについてネットワークの構造を記述し、学習し、テストしてきましたが、次にやれることはたくさんあります。

 * 新しいアーキテクチャを比較する
 * 最適化を調整する、例えば`base_lr`を調整したり単純にもっと長い反復学習を行う
 * ソルバーのアルゴリズムを`SGD`から`AdaDelta`や`Adam`に変更する

上記のようなことを以下のプログラムのテンプレートでいろいろ試行してみてください。`EDIT HERE`と書かれた場所を書き換えてどんどん試しましょう。

以下のプログラムは基準手法として単純な線形識別器を記述しています。

もし何をしたらいいのかよくわからない場合は、例えば

 * 非線形変換を`ReLU`から`ELU`や`Sigmoid`に取り替える
 * 全結合層や非線形変換の層をもっと積み重ねてみる
 * 学習率(`base_lr`)を10倍ずつ変化させてみる(`0.1`や`0.001`を試す)
 * ソルバーを`Adam`に変えてみる(この適応的ソルバーはハイパーパラメータの変動にもう少し頑健なはずですが保証はありません)
 * `niter`を大きくして(例えば500とは1000)、もう少し長い間ソルバーを動かし、学習の違いを観察する

```python
train_net_path = 'mnist/custom_auto_train.prototxt'
test_net_path = 'mnist/custom_auto_test.prototxt'
solver_config_path = 'mnist/custom_auto_solver.prototxt'

### ネットワークの定義
def custom_net(lmdb, batch_size):
    # 自分なりにニューラルネットワークを作ってみましょう
    n = caffe.NetSpec()

    # このデータレイヤーはそのまま使いましょう
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)

    # EDIT HERE このレイヤーは単純な線形識別器を作ります
    n.score =   L.InnerProduct(n.data, num_output=10, weight_filler=dict(type='xavier'))

    # EDIT HERE これは今までに作ったLeNetの一種です
    # n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    # n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    # n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    # n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    # n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    # EDIT HERE これをL.ELUやL.Sigmoidの非線形変換に置き換えてみましょう
    # n.relu1 = L.ReLU(n.fc1, in_place=True)
    # n.score =   L.InnerProduct(n.fc1, num_output=10, weight_filler=dict(type='xavier'))

    # この損失関数はそのまま使いましょう
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()

with open(train_net_path, 'w') as f:
    f.write(str(custom_net('mnist/mnist_train_lmdb', 64)))
with open(test_net_path, 'w') as f:
    f.write(str(custom_net('mnist/mnist_test_lmdb', 100)))

### ソルバーの定義
from caffe.proto import caffe_pb2
s = caffe_pb2.SolverParameter()

# 乱数の初期化、これで再現性のある実験ができます
s.random_seed = 0xCAFFE

# ソルバーにネットワークの定義の場所を指定しておきます
s.train_net = train_net_path
s.test_net.append(test_net_path)
s.test_interval = 500  # 500反復毎にテスト
s.test_iter.append(100) # テストには100バッチを使用

s.max_iter = 10000     # 最大反復回数

# EDIT HERE "SGD"、"Adam"、"Nesterov"といったソルバーを試してみましょう
s.type = "SGD"

# SGE学習率の初期値
s.base_lr = 0.01  # EDIT HERE 変えてみましょう
# モーメンタムは更新量に前回との重み付き平均を取って学習を安定化させる効果があります
s.momentum = 0.9
# 学習減衰率には過学習を防ぐ効果があります
s.weight_decay = 5e-4

# `lr_policy`でどのように学習率を変化させるか決めます
# これはデフォルトのLeNetと同一のものです
s.lr_policy = 'inv'
s.gamma = 0.0001
s.power = 0.75
# EDIT HERE ポリシーを'fixed'にして適応的ソルバーと収束を比較してみましょう
# `fixed`は最も単純に学習率をずっと固定させるものです
# s.lr_policy = 'fixed'

# 現在の損失と正答率を1000反復毎に表示
s.display = 1000

# スナップショットはネットワークの保存先です。今回は5K反復毎、つまり2回経過を保存します
s.snapshot = 5000
s.snapshot_prefix = 'mnist/custom_net'

# CPUで学習します、GPUが使えたら使いましょう
s.solver_mode = caffe_pb2.SolverParameter.CPU

# ソルバーをファイルに保存してファイル名を後で使います
with open(solver_config_path, 'w') as f:
    f.write(str(s))

### ソルバーを読み込んで学習、テスト用のニューラルネットを作ります
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.get_solver(solver_config_path)

### 学習を始めます
niter = 250  # EDIT HERE increase to train for longer
test_interval = niter / 10
# ロスと正答率は記録します
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))

# 学習のメインループです
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # ロスを保存します
    train_loss[it] = solver.net.blobs['loss'].data

    # 指定の間隔で性能評価テスト
    # Caffeではこれを自動でやることもできますが、今回はPythonでやる例
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]))
```
