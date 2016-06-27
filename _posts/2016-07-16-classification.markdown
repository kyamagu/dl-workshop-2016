---
layout: post
title:  "2. Caffeを使った画像分類"
date:   2016-07-16 01:00:00 +09:00
category: caffe
permalink: 2-classification
---

* TOC
{:toc}

ここではCaffeを使って予め学習済みの一般物体認識用のニューラルネットを用い、新しい画像に対して分類を行う流れを学習します。ニューラルネットワークにはA Krizhevskyらが2012年にImageNetデータセットでの分類に提案したもので、Caffeにデフォルトで含まれているものを利用します。

この実習はCaffeのソースに含まれている`examples/00-classification.ipynb`とほぼ同等のものです。

## Caffeを使う準備

最初に必要なライブラリをインポートします。

{% highlight python %}
# numpyとmatplotlibを使うためにインポートします
import numpy as np
import matplotlib.pyplot as plt
# Notebook内でプロットするときにそのままインライン表示するように設定します
%matplotlib inline

# プロットを見やすくするために幾つかディスプレイ設定をしておきます
plt.rcParams['figure.figsize'] = (10, 10)        # 大きめの画像サイズ
plt.rcParams['image.interpolation'] = 'nearest'  # スムージングなし
plt.rcParams['image.cmap'] = 'gray'  # ヒートマップではなくグレースケール表示
{% endhighlight %}

続いてCaffeをインポートします。実習用Docker環境では最初からインポート出来るようになっていますが、その他の環境では`PYTHONPATH`変数を設定するかPython上で`sys.path`を設定する必要があります。

{% highlight python %}
import caffe
{% endhighlight %}

実習環境ではCaffeのソースコードは以下の場所に用意されています

{% highlight python %}
caffe_root = '/home/user/caffe/'
{% endhighlight %}

## 学習済みのネットワークを読み込む

実習環境では予め学習済みのCaffeNetモデルが用意されています。これをファイルから読み込みます。

{% highlight python %}
caffe.set_mode_cpu()            # CPUモードでCaffeを動かします
model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # ニューラルネットの構造を記述したファイルです
                model_weights,  # ニューラルネットのパラメータを保存したファイルです
                caffe.TEST)     # 学習ではなくテストモードでネットワークを作ります
{% endhighlight %}

続いて新しい画像を使うときの前処理方法を記述します。ここでは`caffe.io.Transformer`というCaffeに付属している前処理モジュールを使いますが、用途によっては自前で用意したコードなどで自由に前処理をすることができます。

画像の前処理は新しい画像をニューラルネットワークが受け取ることのできる形に変換する処理です。CaffeNetはデフォルトでBGR配列の画像を受け取ります。また、各ピクセルは`[0, 255]`の範囲の値を取り、ImageNetデータセットの画像のピクセルの平均値を引いた値を入力する前提となっています。更に、画像の色のチャネルは一番最初の次元に格納されているという形式になっています。

Matplotlibを用いるとRGB配置で`[0, 1]`の範囲のピクセルが、列、行、色チャネルの順で格納された配列として画像が読み込まれます。`caffe.io.Transformer`モジュールを使ってこの形式をCaffeNetが受け取れる形式に変換します。

{% highlight python %}
# ImageNetの平均画像を読み込みます（Caffeと一緒に配布されています）
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # BGRの平均ピクセル値を計算します
print 'mean-subtracted values:', zip('BGR', mu)

# 'data'という名前でTransformerを作ります
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # 色チャネルを最初の次元に動かす
transformer.set_mean('data', mu)            # データセットの平均値を引き算
transformer.set_raw_scale('data', 255)      # スケールを[0, 1]から[0, 255]へ
transformer.set_channel_swap('data', (2,1,0))  # RGBからBGRへ
{% endhighlight %}

## CPUを使って画像分類

以上でCaffeNetを使う準備ができました。次に画像をニューラルネットに入力してみます。ここでは1枚の画像しか分類をしませんが、複数枚を同時に処理するためのバッチの大きさを50にしておきます。

{% highlight python %}
# ニューラルネットが受け取る配列の形状を指定します
# デフォルト値でよければやらなくても大丈夫です
net.blobs['data'].reshape(50,        # バッチサイズ
                          3,         # ３色(BGR)画像
                          227, 227)  # 画像の大きさは227x227
{% endhighlight %}

画像をファイルから読み込んで前処理をします。今回はネコの画像を用意してあります。

{% highlight python %}
image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)
{% endhighlight %}

![cat]({{ site.baseurl }}/assets/2-cat.png)
{: style="text-align:center"}

可愛らしいネコの画像は表示されたでしょうか。早速分類してみましょう。

{% highlight python %}
# 画像データをニューラルネットワークのメモリに送ります
net.blobs['data'].data[...] = transformed_image

# ニューラルネットで分類をします
output = net.forward()

output_prob = output['prob'][0]  # バッチの中の最初の行がカテゴリを予測する確率分布のベクトルになっています

print 'predicted class is:', output_prob.argmax()
{% endhighlight %}
`predicted class is: 281`

出力は確率分布がベクトルで表現されたものになっています。このうち最も確率が高いものは281番目のようです。これは正しいカテゴリなのでしょうか？ImageNetデータセットのラベルを少し見てみましょう。この実習環境ではImageNetのカテゴリ名ラベルもあらかじめファイルに用意しておきました。

{% highlight python %}
# load ImageNet labels
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'output label:', labels[output_prob.argmax()]
{% endhighlight %}
`output label: n02123045 tabby, tabby cat`

この画像は`Tabby cat`（ブチ猫）で合ってますね。ついでに他の確率の高そうなカテゴリも見てみましょう。

{% highlight python %}
# Softmax出力をソートしてトップ5の予測を取得
top_inds = output_prob.argsort()[::-1][:5]  # 逆順ソートしてトップ5抜き出し

print 'probabilities and labels:'
zip(output_prob[top_inds], labels[top_inds])
{% endhighlight %}

{% highlight python %}
[(0.31243637, 'n02123045 tabby, tabby cat'),
 (0.2379719, 'n02123159 tiger cat'),
 (0.12387239, 'n02124075 Egyptian cat'),
 (0.10075711, 'n02119022 red fox, Vulpes vulpes'),
 (0.070957087, 'n02127052 lynx, catamount')]
{% endhighlight %}


出てきたカテゴリは`tabby, tabby cat`、`tiger cat`、`Egyptian cat`、`red fox, Vulpes vulpes`、`lynx, catamount`の順です。確率の低いカテゴリの方は惜しい感じのものが出てきていますね。


## ニューラルネットの中間出力

ニューラルネットワークはブラックボックスではありません。少し内部でどのようなことが起きているのか覗いてみましょう。

まず、最初にネットワークがどのような構造になっているのか見てみましょう。畳み込みニューラルネットワークは層構造になっていて、内部のレイヤーにはおおよそ`(batch_size, channel_dim, height, width)`のような4次元状のアクティベーション配列を持っています。このアクティベーションは`OrderedDict`データ型を持っていて`net.blobs`でアクセスできます。入力された画像などのデータを`forward`した時にこの配列の中身にアクティベーションの計算結果が格納されます。次のコードで内部のレイヤーごとのブロブ配列の形を出力してみましょう。

{% highlight python %}
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)
{% endhighlight %}

以下のような出力が出てくると思います。

```
data  (50, 3, 227, 227)
conv1 (50, 96, 55, 55)
pool1 (50, 96, 27, 27)
norm1 (50, 96, 27, 27)
conv2 (50, 256, 27, 27)
pool2 (50, 256, 13, 13)
norm2 (50, 256, 13, 13)
conv3 (50, 384, 13, 13)
conv4 (50, 384, 13, 13)
conv5 (50, 256, 13, 13)
pool5 (50, 256, 6, 6)
fc6 (50, 4096)
fc7 (50, 4096)
fc8 (50, 1000)
prob  (50, 1000)
```

続いて、ネットワークのパラメータの形を見てみましょう。パラメータはに`OrderedDict`データ型として`net.params`からアクセスできます。パラメータにはCNNの重み行列などが格納されています。`net.params`の各レイヤーの要素は`[0]`で重み、`[1]`でバイアスにアクセスできるようになっています。パラメータの形は重みは`(output_channels, input_channels, filter_height, filter_width)`といった4次元状の配列でバイアスは`(output_channels,)`といった1次元の配列として格納されていることがほとんどです。次のコードで形を出力してみましょう。

{% highlight python %}
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
{% endhighlight %}

```
conv1 (96, 3, 11, 11) (96,)
conv2 (256, 48, 5, 5) (256,)
conv3 (384, 256, 3, 3) (384,)
conv4 (384, 192, 3, 3) (384,)
conv5 (256, 192, 3, 3) (256,)
fc6 (4096, 9216) (4096,)
fc7 (4096, 4096) (4096,)
fc8 (1000, 4096) (1000,)
```

ここでは4次元の配列を扱っているので、ヒートマップを使って可視化する関数を用意してみましょう。

{% highlight python %}
def vis_square(data):
    """形が (n, height, width) か (n, height, width, 3) の配列を受け取り
      (height, width) をおおよそ sqrt(n) by sqrt(n) のグリッドに表示"""

    # まずはデータを正規化
    data = (data - data.min()) / (data.max() - data.min())

    # 可視化するフィルタの数を正方形に揃える
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # フィルタの間に隙間を挿入
               + ((0, 0),) * (data.ndim - 3))  # 最後は隙間なし
    data = np.pad(data, padding, mode='constant', constant_values=1)  # 隙間は白

    # フィルタをタイル状に並べて画像化する
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data); plt.axis('off')
{% endhighlight %}

最初に`conv1`レイヤーのパラメータを可視化してみましょう。

{% highlight python %}
# パラメータは [weights, biases] のリストです
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))
{% endhighlight %}

![conv1-filters]({{ site.baseurl }}/assets/2-conv1-filters.png)
{: style="text-align: center"}

続いて`conv1`レイヤーのアクティベーションのうち、最初の36チャネル分を見てみましょう。

{% highlight python %}
feat = net.blobs['conv1'].data[0, :36]
vis_square(feat)
{% endhighlight %}

![conv1-activations]({{ site.baseurl }}/assets/2-conv1-activations.png)
{: style="text-align: center"}

`pool5`レイヤーの後ではアクティベーションは次のようになります。

{% highlight python %}
feat = net.blobs['pool5'].data[0]
vis_square(feat)
{% endhighlight %}

![pool5-activations]({{ site.baseurl }}/assets/2-pool5-activations.png)
{: style="text-align: center"}

最初の全結合層`fc6`の様子はどうでしょうか。ここでは出力をヒストグラムとして表示することにします。

{% highlight python %}
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
{% endhighlight %}

![fc6-activations]({{ site.baseurl }}/assets/2-fc6-activations.png)
{: style="text-align: center"}

最後に、Softmax関数の出力`prob`ブロブからカテゴリごとの確率分布を表示してみましょう。

{% highlight python %}
feat = net.blobs['prob'].data[0]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)
{% endhighlight %}

![fc8-activations]({{ site.baseurl }}/assets/2-fc8-activations.png)
{: style="text-align: center"}

ここでは一部のクラスタで強く出力が出ているのがわかります。ラベルはカテゴリの近い順序で並べてあるのでこのようにネコ科のところで強く集まって出力が見えるわけです。ピークになっている場所が、上で見たように一番確率の大きいカテゴリです。

## 自分の画像を分類してみる

一通り画像の分類方法を見てきたので新しくWebから持ってきた画像を分類してみましょう。以下のコードで`my_image_url`を好きなJPEG画像のURLに置き換えて動かしてみましょう。

{% highlight python %}
# 画像をダウンロード
my_image_url = "..."  # ここに画像のURLを指定
# 例えば
# my_image_url = "https://upload.wikimedia.org/wikipedia/commons/b/be/Orang_Utan%2C_Semenggok_Forest_Reserve%2C_Sarawak%2C_Borneo%2C_Malaysia.JPG"
!wget -O image.jpg $my_image_url

# 画像を読み込んでネットワークにコピー
image = caffe.io.load_image('image.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', image)

# 分類処理
net.forward()

# 出力される確率分布を取得
output_prob = net.blobs['prob'].data[0]

# Softmax出力のうちトップ5のカテゴリを取得
top_inds = output_prob.argsort()[::-1][:5]

plt.imshow(image)

print 'probabilities and labels:'
zip(output_prob[top_inds], labels[top_inds])
{% endhighlight %}
