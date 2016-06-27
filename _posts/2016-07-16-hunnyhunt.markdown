---
layout: post
title:  "4. 学習済みのネットワークをマルハナバチ分類にファインチューニング"
date:   2016-07-16 04:00:00 +09:00
category: caffe
permalink: 4-hunnyhunt
---

このセクションではディープラーニングモデルを実応用で使うのに特に役立つファインチューニングを試してみましょう。ファインチューニングは予め他のデータセットで学習されたネットワークのパラメータを新しいデータに転移して使うやり方です。

この方法の利点は、ネットワークが予め大きなデータセットで学習してあるので、中間層では最初からすでに一般的な視覚的な刺激に対して十分なセマンティクスを抜き出せると期待できる点です。強力な特徴量表現を抽出するブラックボックスのようなものだと思ってください。この上に数層を重ねることで新しいデータでも良い識別性能を発揮できるようにします。

このセクションはおおよそ`caffe/examples/02-fine-tuning.ipynb`に従っていますが、マルハナバチのデータを使用する点で少し中身が異なっています。

なお、このデモはGPUありのCaffe環境を前提としています。CPUのみでやる場合は計算時間が膨大になるので十分なメモリを持ったGPUとCUDA環境を用意して実行することをお勧めします。

## 1. セットアップとデータの準備

最初にパスを設定し、必要なモジュールをインポートしましょう。


```python
caffe_root = '/home/user/caffe/'  # このデモは次のパスで実行 {caffe_root}/examples (そうでなければここを変えましょう)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

import numpy as np
from pylab import *
%matplotlib inline
import tempfile
```

まずはデータを準備する必要があります。これには以下の手順が必要です。

 1. ImageNetで学習済みのモデルをダウンロード
 2. このデモのためのミツバチのデータセットをダウンロード


```python
import os
os.chdir(caffe_root)

# まずはImageNet学習済みモデルを入手
if not os.path.exists('data/ilsvrc12/imagenet_mean.binaryproto'):
    !data/ilsvrc12/get_ilsvrc_aux.sh
    !scripts/download_model_binary.py models/bvlc_reference_caffenet

# 続いてミツバチのデータをダウンロード
if not os.path.exists("data/bees"):
    !wget http://vision.is.tohoku.ac.jp/hunnyhunt/static/data/bees.tgz -O data/bees.tgz
    !tar xzf data/bees.tgz
```

ちゃんと学習済みネットワークがダウンロードされているか確認しましょう。


```python
weights = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
assert os.path.exists(weights)
```

今回扱うミツバチのデータはマルハナバチを中心に15種類のミツバチに集めたものです。データは`data/bees`に展開されています。少し中身を確認してみましょう。


```python
!ls data/bees
```

    a.cerana       b.deuteronymus		     b.hypocrita   labels.txt
    a.mellifera    b.deuteronymus.maruhanabachi  b.ignitus	   test.txt
    b.ardens       b.diversus		     b.shrencki    train.txt
    b.beaticola    b.honshuensis		     b.terrestris  val.txt
    b.consobrinus  b.hypnorum		     b.yezoensis


中にはテキストファイルと種別毎にフォルダに分けられた画像が入っています。画像は例えば以下のようにフォルダに格納されています。


```python
!ls data/bees/b.deuteronymus/
```

    35694_0.jpg  46829_0.jpg  56100_0.jpg  56178_0.jpg  57010_0.jpg  57638_0.jpg
    36884_0.jpg  47598_0.jpg  56124_0.jpg  56290_0.jpg  57162_0.jpg
    39123_0.jpg  55694_0.jpg  56125_0.jpg  57006_0.jpg  57177_0.jpg
    39124_0.jpg  56098_0.jpg  56176_0.jpg  57008_0.jpg  57636_0.jpg
    39127_0.jpg  56099_0.jpg  56177_0.jpg  57009_0.jpg  57637_0.jpg


データセットに含まれるミツバチのラベルは`labels.txt`ファイルに、学習(train)、テスト(test)、検証(validation)のためのサンプルの分割は`train.txt`、`test.txt`、`val.txt`ファイルにそれぞれテキスト形式で格納されています。これはCaffeで`ImageDataLayer`を使って画像を入力するときに使うフォーマットです。


```python
!cat data/bees/labels.txt
```

    a.cerana
    a.mellifera
    b.ardens
    b.beaticola
    b.consobrinus
    b.deuteronymus
    b.deuteronymus.maruhanabachi
    b.diversus
    b.honshuensis
    b.hypnorum
    b.hypocrita
    b.ignitus
    b.shrencki
    b.terrestris
    b.yezoensis



```python
!head data/bees/train.txt
```

    data/bees/b.ardens/43061_0.jpg 2
    data/bees/b.ignitus/46592_0.jpg 11
    data/bees/b.ignitus/46442_0.jpg 11
    data/bees/b.diversus/47554_0.jpg 7
    data/bees/b.hypocrita/39267_0.jpg 10
    data/bees/b.diversus/44665_0.jpg 7
    data/bees/b.ardens/50148_0.jpg 2
    data/bees/b.ardens/45554_0.jpg 2
    data/bees/b.ardens/46340_0.jpg 2
    data/bees/b.beaticola/46839_0.jpg 3


ImageNetの1000カテゴリのラベルを`ilsvrc12/synset_words.txt`から、そしてミツバチの種別ラベルを `data/bees/labels.txt`からPythonに読み込みましょう。


```python
# ImageNetのカテゴリをimagenet_labelsに読み込み
imagenet_label_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
imagenet_labels = list(np.loadtxt(imagenet_label_file, str, delimiter='\t'))
assert len(imagenet_labels) == 1000
print 'Loaded ImageNet labels:\n', '\n'.join(imagenet_labels[:10] + ['...'])

# ミツバチのカテゴリをbee_labelsに読み込み
bee_label_file = caffe_root + 'data/bees/labels.txt'
bee_labels = list(np.loadtxt(bee_label_file, str, delimiter='\n'))
print '\nLoaded bee labels (' + str(len(bee_labels)) + '):\n', ', '.join(bee_labels)
```

    Loaded ImageNet labels:
    n01440764 tench, Tinca tinca
    n01443537 goldfish, Carassius auratus
    n01484850 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
    n01491361 tiger shark, Galeocerdo cuvieri
    n01494475 hammerhead, hammerhead shark
    n01496331 electric ray, crampfish, numbfish, torpedo
    n01498041 stingray
    n01514668 cock
    n01514859 hen
    n01518878 ostrich, Struthio camelus
    ...

    Loaded bee labels (15):
    a.cerana, a.mellifera, b.ardens, b.beaticola, b.consobrinus, b.deuteronymus, b.deuteronymus.maruhanabachi, b.diversus, b.honshuensis, b.hypnorum, b.hypocrita, b.ignitus, b.shrencki, b.terrestris, b.yezoensis


## 2. ネットワークを定義する

最初は`Caffenet`の構造を記述するところから始めましょう。`CaffeNet`は`AlexNet`を少しだけ改良した畳み込みニューラルネットで、以下の`caffenet`関数は与えられたデータとクラス数に対してネットワーク構造を初期化します。


```python
from caffe import layers as L
from caffe import params as P

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    # write the net to a temporary file and return its filename
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(n.to_proto()))
        return f.name
```

CaffeNetをダミーのデータを入力として作り、外部からの入力画像を使って分類ができるようにします。


```python
dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))
imagenet_net_filename = caffenet(data=dummy_data, train=False)
imagenet_net = caffe.Net(imagenet_net_filename, weights, caffe.TEST)
```

続いて`bee_net`関数を定義します。これは内部的に`caffenet`関数を呼び、ミツバチデータに対してネットワークを作ります。新しいネットワークは同じCaffeNetアーキテクチャですが、入力と出力が異なります。

 * 入力はミツバチデータを使います。これをネットワークの`ImageData`レイヤーに入力します。
 * 出力は元のImageNetの1000カテゴリではなくミツバチ15カテゴリです。
 * 識別器のレイヤーは`fc8`から`fc8_bee`に名前を変更します。さもなくばネットワークを読み込んだ時に元の1000カテゴリ用の`fc8`のパラメータを読んでしまうためです。


```python
def bee_net(train=True, learn_all=False, subset=None, num_classes=15):
    if subset is None:
        subset = 'train' if train else 'test'
    source = caffe_root + 'data/bees/%s.txt' % subset
    transform_param = dict(mirror=train, crop_size=227,
        mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
    style_data, style_label = L.ImageData(
        transform_param=transform_param, source=source,
        batch_size=50, new_height=256, new_width=256, ntop=2)
    return caffenet(data=style_data, label=style_label, train=train,
                    num_classes=15,
                    classifier_name='fc8_bee',
                    learn_all=learn_all)
```

ここで定義した`bee_net`関数を使って`untrained_bee_net`を初期化します。これはCaffeNetの改造版でミツバチのデータセットから入力を取得し、ImageNetからパラメータを転移してきたものです。

`forward`メソッドを使って`untrained_bee_net`に学習データからバッチを受け取りましょう。


```python
untrained_bee_net = caffe.Net(bee_net(train=False, subset='train'),
                              weights, caffe.TEST)
untrained_bee_net.forward()
bee_data_batch = untrained_bee_net.blobs['data'].data.copy()
bee_label_batch = np.array(untrained_bee_net.blobs['label'].data, dtype=np.int32)
```

サイズ50のバッチからミツバチの画像を抜き出してみましょう（今回は適当に8番目の画像）。画像を表示して、それから`imagenet_net`を通し、ImageNetで学習済みネットワークが予測する1000カテゴリのうちの最もらしい5つのカテゴリを出力してみます。画面表示のためのヘルパー関数を幾つか以下に用意しました。


```python
def disp_preds(net, image, labels, k=5, name='ImageNet'):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]
    top_k = (-probs).argsort()[:k]
    print 'top %d predicted %s labels =' % (k, name)
    print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                    for i, p in enumerate(top_k))

def disp_imagenet_preds(net, image):
    disp_preds(net, image, imagenet_labels, name='ImageNet')

def disp_bee_preds(net, image):
    disp_preds(net, image, bee_labels, name='bee')

# 前処理された画像を元に戻すためのヘルパー関数、表示に便利
def deprocess_net_image(image):
    image = image.copy()              # 元の画像を変更しないように
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # おおよその平均値加算

    # 値を [0, 255] に封じ込める
    image[image < 0], image[image > 255] = 0, 255

    # float32 から uint8 へ
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image
```

以下は元の画像と正解ラベルです。


```python
batch_index = 8
image = bee_data_batch[batch_index]
plt.imshow(deprocess_net_image(image))
print 'actual label =', bee_labels[bee_label_batch[batch_index]]
```

    actual label = b.ardens



![png]({{ site.baseurl }}/assets/4-hunnyhunt/output_27_1.png)


次にImageNetの予測そのままの場合


```python
disp_imagenet_preds(imagenet_net, image)
```

    top 5 predicted ImageNet labels =
    	(1) 75.08% n02206856 bee
    	(2)  9.74% n02493509 titi, titi monkey
    	(3)  5.88% n02494079 squirrel monkey, Saimiri sciureus
    	(4)  2.66% n02493793 spider monkey, Ateles geoffroyi
    	(5)  1.36% n02492035 capuchin, ringtail, Cebus capucinus


そして新しい全く学習していない`bee_net`の場合


```python
disp_bee_preds(untrained_bee_net, image)
```

    top 5 predicted bee labels =
    	(1)  6.67% a.cerana
    	(2)  6.67% a.mellifera
    	(3)  6.67% b.ardens
    	(4)  6.67% b.beaticola
    	(5)  6.67% b.consobrinus


今回はたまたま元のImageNetの予測にちゃんと`bee`と出てきました。他の画像の場合はそんなに上手くいかないかもしれませんし、もっと言えばImageNetのカテゴリにはミツバチの細かな分類カテゴリはないので正解の`b.ardens`は絶対に出てきません。`batch_index`を8以外で0から49の間の数字に変えて予測がどうなるか試してみましょう。このバッチの50枚以上の画像を見るにはさらに`forward`をしてバッチを入れ替えます。

さて、今回はImageNet学習済みモデルと`bee_net`で`conv1`から`fc7`層までが全く同じです。これが本当かどうか、`fc7`の出力を比べることで以下のように確認することができます。


```python
diff = untrained_bee_net.blobs['fc7'].data[0] - imagenet_net.blobs['fc7'].data[0]
error = (diff ** 2).sum()
assert error < 1e-8
```

終わったら`untrained_bee_net`を破棄してメモリを解放しましょう。`imagenet_net`はまた後で使います。


```python
del untrained_bee_net
```

## 3. ミツバチ識別器の学習

それでは`solver`関数を定義し、その中でCaffeのソルバーを作りましょう。この関数では学習やスナップショット取得に関する様々なパラメータを定義します。それぞれの意味はコメント行を見てください。学習結果を改善するためにいろいろと設定を変更してみるといいでしょう。


```python
from caffe.proto import caffe_pb2

def solver(train_net_path, test_net_path=None, base_lr=0.001):
    s = caffe_pb2.SolverParameter()

    # 学習とテスト用のネットワークのパス
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # テストは1000反復毎
        s.test_iter.append(100) # テスト時は100バッチ使用

    # 勾配を平滑化するのに使う反復回数
    # これを使うとメモリを使うことなくバッチサイズを与えられた値の倍数にするのと等価な効果
    s.iter_size = 1

    s.max_iter = 100000     # 学習の反復回数

    # 確率的勾配効果法(SGD)を使う。他には'Adam'、'RMSProp'など指定可
    s.type = 'SGD'

    # 学習率の初期値
    s.base_lr = base_lr

    # `lr_policy`を使ってどのように学習率を変化させていくかを決める
    # 'step'では'stepsize'毎に学習率を'gamma'倍する
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 20000

    # SGDのハイパーパラメータの指定
    # `momentum`は現在の勾配と過去の勾配の重み付き平均の重みで、これが大きいと学習が安定
    # L2重み減衰は学習を正則化し、過学習を防ぐ
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # 現在の損失(loss)と正答率を1000反復毎に表示
    s.display = 1000

    # スナップショットは学習しているネットワークの保存先ファイル。今回は10K反復毎に
    # スナップショットを作成
    s.snapshot = 10000
    s.snapshot_prefix = caffe_root + 'models/finetune_bees/finetune_bees'

    # GPUで学習。CPUは学習するには遅すぎるので止めましょう
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    # 以上のソルバーの設定は一時ファイルに保存します
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        return f.name

# 学習したスナップショットを置く場所を作っておきます。
if not os.path.exists(caffe_root + 'models/finetune_bees/'):
    os.makedirs(caffe_root + 'models/finetune_bees/')
```

さて、ソルバーの準備ができたので学習を始めてみましょう。

ちなみにですが、UNIXコマンドラインでニューラルネットワークの学習をするときはこんな風にコマンド入力します。

```bash
build/tools/caffe train \
    -solver models/finetune_bee/solver.prototxt \
    -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
    -gpu 0
```

今回はPythonを使います。

最初に`run_solver`関数を作ります。この関数はソルバーのリストを受け取り、それぞれのソルバーを順繰りにステップします。また、その時の反復毎の正答率と損失を記録します。終わったら学習したパラメータをファイルに書き出します。


```python
def run_solvers(niter, solvers, disp_interval=10):
    """ソルバーをniter回反復させ、記録した損失と正答率を返す。
       `solvers`は (name, solver) のタプルのリスト"""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # SGEステップを一回進める
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)
    # 両方のネットからパラメータを保存
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights
```

ソルバーを作って実際に動かしてみましょう。今回は二つのソルバーを作ります。一つは`bee_solver`でImageNetから学習済みのパラメータを転移させたもの（`copy_from`でやります）、もう一つは`scratch_bee_solver`でパラメータをランダムに初期化したものを学習します。

学習中はImageNetから転移したものの方がより早く収束してするのが見えると思います。


```python
niter = 200  # number of iterations to train

# Reset style_solver as before.
bee_solver_filename = solver(bee_net(train=True))
bee_solver = caffe.get_solver(bee_solver_filename)
bee_solver.net.copy_from(weights)

# For reference, we also create a solver that isn't initialized from
# the pretrained ImageNet weights.
scratch_bee_solver_filename = solver(bee_net(train=True))
scratch_bee_solver = caffe.get_solver(scratch_bee_solver_filename)

print 'Running solvers for %d iterations...' % niter
solvers = [('pretrained', bee_solver),
           ('scratch', scratch_bee_solver)]
loss, acc, weights = run_solvers(niter, solvers)
print 'Done.'

train_loss, scratch_train_loss = loss['pretrained'], loss['scratch']
train_acc, scratch_train_acc = acc['pretrained'], acc['scratch']
bee_weights, scratch_bee_weights = weights['pretrained'], weights['scratch']

# Delete solvers to save memory.
del bee_solver, scratch_bee_solver, solvers
```

    Running solvers for 200 iterations...
      0) pretrained: loss=2.708, acc= 0%; scratch: loss=2.708, acc= 0%
     10) pretrained: loss=1.912, acc=50%; scratch: loss=2.530, acc=24%
     20) pretrained: loss=1.755, acc=46%; scratch: loss=2.293, acc=30%
     30) pretrained: loss=1.845, acc=44%; scratch: loss=2.286, acc=16%
     40) pretrained: loss=1.879, acc=46%; scratch: loss=2.292, acc=32%
     50) pretrained: loss=0.973, acc=64%; scratch: loss=2.293, acc=24%
     60) pretrained: loss=1.506, acc=52%; scratch: loss=2.196, acc=24%
     70) pretrained: loss=1.382, acc=52%; scratch: loss=2.321, acc=24%
     80) pretrained: loss=1.441, acc=58%; scratch: loss=2.082, acc=34%
     90) pretrained: loss=1.165, acc=58%; scratch: loss=2.060, acc=18%
    100) pretrained: loss=1.053, acc=72%; scratch: loss=1.779, acc=44%
    110) pretrained: loss=1.098, acc=56%; scratch: loss=2.023, acc=30%
    120) pretrained: loss=1.576, acc=56%; scratch: loss=2.120, acc=20%
    130) pretrained: loss=0.998, acc=58%; scratch: loss=2.022, acc=32%
    140) pretrained: loss=1.093, acc=68%; scratch: loss=2.245, acc=20%
    150) pretrained: loss=1.002, acc=62%; scratch: loss=2.049, acc=28%
    160) pretrained: loss=1.363, acc=56%; scratch: loss=2.296, acc=20%
    170) pretrained: loss=1.541, acc=54%; scratch: loss=2.236, acc=26%
    180) pretrained: loss=1.086, acc=54%; scratch: loss=2.191, acc=26%
    190) pretrained: loss=1.132, acc=58%; scratch: loss=2.048, acc=14%
    199) pretrained: loss=1.451, acc=54%; scratch: loss=2.156, acc=22%
    Done.


二つの学習プロセスの正答率と損失を観察してみましょう。ImageNet学習済みモデル（青線）がより速く収束し、ランダム初期化モデル（緑線）はほとんど識別性能が変わっていないことに注目してください。


```python
plot(np.vstack([train_loss, scratch_train_loss]).T)
xlabel('Iteration #')
ylabel('Loss')
```




    <matplotlib.text.Text at 0x7fdca883e290>




![png]({{ site.baseurl }}/assets/4-hunnyhunt/output_43_1.png)



```python
plot(np.vstack([train_acc, scratch_train_acc]).T)
xlabel('Iteration #')
ylabel('Accuracy')
```




    <matplotlib.text.Text at 0x7fdca8883190>




![png]({{ site.baseurl }}/assets/4-hunnyhunt/output_44_1.png)


200反復後のテストデータのの正答率を見てみましょう。今回は15カテゴリから予測をしているので、ランダム予測のチャンスは6.67%ほどです。今回は両方のネットワークでもこれよりも良い正答率になり、さらにImageNetで初期化したネットワークの方がランダム初期化のものよりも良い性能になることが期待されます。見てみましょう。


```python
def eval_bee_net(weights, test_iters=10):
    test_net = caffe.Net(bee_net(train=False), weights, caffe.TEST)
    accuracy = 0
    for it in xrange(test_iters):
        accuracy += test_net.forward()['acc']
    accuracy /= test_iters
    return test_net, accuracy
```


```python
test_net, accuracy = eval_bee_net(bee_weights)
print 'Accuracy, trained from ImageNet initialization: %3.1f%%' % (100*accuracy, )
scratch_test_net, scratch_accuracy = eval_bee_net(scratch_bee_weights)
print 'Accuracy, trained from   random initialization: %3.1f%%' % (100*scratch_accuracy, )
```

    Accuracy, trained from ImageNet initialization: 57.8%
    Accuracy, trained from   random initialization: 24.6%


## 4. End-to-endのファインチューニング

この節では先ほど学習した両方のネットワークを再び学習します。今回は画像に直接適用する`conv1`フィルタから予測までの全てのレイヤーをend-to-endで学習します。今回は`learn_all=True`の引数を`bee_net`に与えることで非ゼロの学習率係数`lr_mult`を全てのパラメータに適用します。デフォルトは`learn_all=False`としてあったので、`conv1`から`fc7`までのパラメータは固定され(`lr_mult=0`)、最終の`fc8_bee`レイヤーのみが学習されるようになっていました。

今回は両方のネットワークは先ほどの学習セッションの最終正答率の状態から学習を始め、end-to-endの学習によって大きく性能を引き上げます。もう少し厳密に測定するならば、end-to-endの学習なしでも同じようにこの状態から学習を続け、同一の反復回数でどの程度の性能差が出るのか観察してみるのも面白いでしょう。ぜひやってみてください。


```python
end_to_end_net = bee_net(train=True, learn_all=True)

# 学習率の初期値を前回同様1e-3にします
# この値や他の最適化パラメータの調節はいろいろ試してみてください。
# 例えば、損失がinfinityやNaNに発散してしまう場合、`base_lr`を1e-4、1e-5と下げてみて
# 発散しなくなる値を探ると良いでしょう
base_lr = 0.001

bee_solver_filename = solver(end_to_end_net, base_lr=base_lr)
bee_solver = caffe.get_solver(bee_solver_filename)
bee_solver.net.copy_from(bee_weights)

scratch_bee_solver_filename = solver(end_to_end_net, base_lr=base_lr)
scratch_bee_solver = caffe.get_solver(scratch_bee_solver_filename)
scratch_bee_solver.net.copy_from(scratch_bee_weights)

print 'Running solvers for %d iterations...' % niter
solvers = [('pretrained, end-to-end', bee_solver),
           ('scratch, end-to-end', scratch_bee_solver)]
_, _, finetuned_weights = run_solvers(niter, solvers)
print 'Done.'

bee_weights_ft = finetuned_weights['pretrained, end-to-end']
scratch_bee_weights_ft = finetuned_weights['scratch, end-to-end']

# 終わったらメモリを解放しましょう
del bee_solver, scratch_bee_solver, solvers
```

    Running solvers for 200 iterations...
      0) pretrained, end-to-end: loss=1.129, acc=60%; scratch, end-to-end: loss=1.920, acc=30%
     10) pretrained, end-to-end: loss=1.300, acc=54%; scratch, end-to-end: loss=2.171, acc=24%
     20) pretrained, end-to-end: loss=1.148, acc=62%; scratch, end-to-end: loss=2.189, acc=30%
     30) pretrained, end-to-end: loss=1.418, acc=48%; scratch, end-to-end: loss=2.232, acc=16%
     40) pretrained, end-to-end: loss=1.333, acc=54%; scratch, end-to-end: loss=2.296, acc=32%
     50) pretrained, end-to-end: loss=0.829, acc=72%; scratch, end-to-end: loss=2.206, acc=24%
     60) pretrained, end-to-end: loss=1.068, acc=64%; scratch, end-to-end: loss=2.135, acc=24%
     70) pretrained, end-to-end: loss=0.990, acc=70%; scratch, end-to-end: loss=2.289, acc=24%
     80) pretrained, end-to-end: loss=0.877, acc=70%; scratch, end-to-end: loss=2.081, acc=34%
     90) pretrained, end-to-end: loss=0.755, acc=74%; scratch, end-to-end: loss=2.026, acc=18%
    100) pretrained, end-to-end: loss=0.668, acc=80%; scratch, end-to-end: loss=1.754, acc=44%
    110) pretrained, end-to-end: loss=0.728, acc=80%; scratch, end-to-end: loss=2.007, acc=30%
    120) pretrained, end-to-end: loss=0.806, acc=70%; scratch, end-to-end: loss=2.085, acc=18%
    130) pretrained, end-to-end: loss=0.508, acc=82%; scratch, end-to-end: loss=2.001, acc=32%
    140) pretrained, end-to-end: loss=0.744, acc=78%; scratch, end-to-end: loss=2.225, acc=26%
    150) pretrained, end-to-end: loss=0.843, acc=70%; scratch, end-to-end: loss=2.012, acc=28%
    160) pretrained, end-to-end: loss=0.989, acc=64%; scratch, end-to-end: loss=2.279, acc=24%
    170) pretrained, end-to-end: loss=0.592, acc=80%; scratch, end-to-end: loss=2.219, acc=28%
    180) pretrained, end-to-end: loss=0.592, acc=80%; scratch, end-to-end: loss=2.154, acc=36%
    190) pretrained, end-to-end: loss=0.415, acc=86%; scratch, end-to-end: loss=2.005, acc=24%
    199) pretrained, end-to-end: loss=0.581, acc=78%; scratch, end-to-end: loss=2.069, acc=40%
    Done.


End-to-endの学習が終わったら早速試してみましょう。今回は全てのレイヤーがミツバチ分類に最適化されているので、両方のネットワークともに前回の最終識別層のみの学習のときよりもさらに良い性能になっているはずです。


```python
test_net, accuracy = eval_bee_net(bee_weights_ft)
print 'Accuracy, finetuned from ImageNet initialization: %3.1f%%' % (100*accuracy, )
scratch_test_net, scratch_accuracy = eval_bee_net(scratch_bee_weights_ft)
print 'Accuracy, finetuned from   random initialization: %3.1f%%' % (100*scratch_accuracy, )
```

    Accuracy, finetuned from ImageNet initialization: 64.8%
    Accuracy, finetuned from   random initialization: 36.2%


最初に使った画像で識別してみましょう。


```python
plt.imshow(deprocess_net_image(image))
disp_bee_preds(test_net, image)
```

    top 5 predicted bee labels =
    	(1) 48.27% b.ardens
    	(2) 11.24% a.mellifera
    	(3)  9.78% b.consobrinus
    	(4)  6.97% b.honshuensis
    	(5)  5.49% b.diversus



![png]({{ site.baseurl }}/assets/4-hunnyhunt/output_53_1.png)


今回はだいぶ違いが出てきているようですね。ただ、この画像は学習用データにも含まれるものなので学習時にネットワークはこの画像を見たことがあるはずです。

次に学習に使っていないテスト用画像から画像を持ってきてend-to-endのファインチューンネットワークがどのように予測をするか見てみましょう。


```python
batch_index = 1
image = test_net.blobs['data'].data[batch_index]
plt.imshow(deprocess_net_image(image))
print 'actual label =', bee_labels[int(test_net.blobs['label'].data[batch_index])]
```

    actual label = b.diversus



![png]({{ site.baseurl }}/assets/4-hunnyhunt/output_55_1.png)



```python
disp_bee_preds(test_net, image)
```

    top 5 predicted bee labels =
    	(1) 99.66% b.diversus
    	(2)  0.11% b.hypnorum
    	(3)  0.09% b.honshuensis
    	(4)  0.05% b.ignitus
    	(5)  0.04% a.mellifera


スクラッチ（ランダム初期化）から学習したネットワークの予測も見てみましょう。スクラッチのネットワークも2番目に正しいカテゴリを予測していますが、かなり他のカテゴリと混乱しているようです。


```python
disp_bee_preds(scratch_test_net, image)
```

    top 5 predicted bee labels =
    	(1) 29.28% b.ardens
    	(2) 25.10% b.diversus
    	(3) 12.26% b.hypocrita
    	(4)  9.24% b.ignitus
    	(5)  5.71% a.mellifera


もちろん、ImageNetモデルの予測と比較することができます


```python
disp_imagenet_preds(imagenet_net, image)
```

    top 5 predicted ImageNet labels =
    	(1) 34.88% n02206856 bee
    	(2) 28.83% n01774750 tarantula
    	(3)  2.40% n01775062 wolf spider, hunting spider
    	(4)  2.11% n02493793 spider monkey, Ateles geoffroyi
    	(5)  1.72% n02107312 miniature pinscher


これまでファインチューニングのやり方を見てきました。あとはより大きなデータセットで長い間反復学習を行った時にどのように予測が行われるか試行してみてください。
