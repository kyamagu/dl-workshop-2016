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

この画像は`Tabby cat`で合ってますね。ついでに他の確率の高そうなカテゴリも見てみましょう。

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


