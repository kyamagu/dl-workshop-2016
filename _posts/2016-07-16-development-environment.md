---
layout: post
title:  "5. Caffeの動作環境に関して"
date:   2016-07-16 03:00:00 +09:00
category: misc
permalink: 5-development-environment
---

* TOC
{:toc}

## Caffeについて

CaffeはYangqing Jiaらが中心となってBerkeley Vision and Learning Center (BVLC)で開発を始めたディープラーニング用のフレームワークです。特徴としては本体がC++で開発されており、GPUを使うことで高速に学習を行うことができる点、PythonやMatlabからの利用も可能な点、研究コミュニティでの利用者が多くコミュニティサイトを通して学習済みモデルの配布が行なわれている点、オープンソースでコミュニティ中心のコード開発が進んでいる点などが挙げられます。

CaffeはGithubの以下URLにホストされています。

[https://github.com/BVLC/caffe](https://github.com/BVLC/caffe)

## Caffeのビルド

### Caffeのビルドの依存関係

最新のインストール手順は[公式ドキュメント](http://caffe.berkeleyvision.org/installation.html)を参照してください。Caffeは基本的にUNIX環境を念頭に置いて開発され、公式にはUbuntu 16.04-12.04、Mac OS 10.11-10.8、Docker、AWS環境での動作確認が行われています。特にUbuntu Linux環境が標準的な動作プラットフォームになっています。

ビルドには幾つかの依存関係のライブラリをあらかじめ用意しておく必要があります。

 * [CUDA](https://developer.nvidia.com/cuda-zone)はGPUを使うために必須です
   * Version 7+と最新のドライバが推奨されます
   * Version 5.5以前は古いので非推奨
 * [BLAS](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)はATLAS、MKL、OpenBLASといった実装が使えます
 * [Boost](http://www.boost.org/)はVersion 1.55以上
 * `protobuf`、`glog`、`gflags`、`hdf5`

必須ではありませんがオプションとして

 * [OpenCV](http://opencv.org/) >= 2.4または3.0以降
 * IO関連: `lmdb`、`leveldb`
 * cuDNN v5: GPUの高速演算

PyCaffeとMatcaffeラッパーはそれぞれの依存関係があります

 * Python: `Python 2.7`または`Python 3.3+`、`numpy >= 1.7`、`boost.python`
 * MATLAB: `mex`コンパイラー

**cuDNN Caffe**: Caffeは[NVIDIA cuDNN](https://developer.nvidia.com/cudnn)を使うとGPUの演算速度が最も速くなります。cuDNNをインストールして`Makefile.config`中で`USE_CUDNN := 1`というフラグをつけてビルドすると使えるようになります。

**CPU-only Caffe**: GPUを使わずにCPUだけで使う場合は`CPU_ONLY := 1`というフラグを`Makefile.config`で指定します。

### Ubuntu 14.04 LTSでのインストール手順

必要なパッケージは`apt-get`で揃えることができます。

```bash
sudo apt-get install build-essential git wget libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install --no-install-recommends libboost-all-dev
```

GPUを使う場合は[NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)をダウンロードしてインストールする必要があります。ダウンロードしてきたパッケージは以下のようにしてインストールできます。インストール後は再起動が必要かもしれません。GPUが動作しているかを確認するには`nvidia-smi`コマンドを使うといいです。

```bash
sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

GPUをディープラーニングに使うにはメモリ容量のなるべく大きなものが望ましいです。例えばNVIDIA GTX Titan-XやGTX 1080など、なるべくハイエンドなものを使いましょう。

cuDNNを使う場合は更に[NVIDIA cuDNN](https://developer.nvidia.com/cudnn)をダウンロードする必要があります。こちらは現在無料の開発者メンバー登録が必要です。ダウンロードしてきたcuDNNパッケージは以下のようにシステムにインストールします。

```bash
tar xzf cudnn-7.5-linux-x64-v5.0-rc.tgz
sudo cp cuda/include/* /usr/local/cuda/include/
sudo cp cuda/lib64/* /usr/local/cuda/lib64/
```

Caffeのソースコードは`git`でダウンロードします。

```bash
git clone https://github.com/BVLC/caffe.git
```

ビルドは`make`を使うのが標準的です。`make`する前に`Makefile.config`ファイルを作成し、ビルドの様々なオプションを指定します。ビルドにはCMakeを使う方法もあります。

```bash
cd caffe/
cp Makefile.config.example Makefile.config
```

この`Makefile.config`ファイルをテキストエディタで開き、内容を編集します。例えば`USE_CUDNN := 1`を指定してcuDNNを使えるようにしたり、Anaconda Pythonを使っている人はそれに対応する項目を変更したり、プラットフォームに応じて編集をしてください。終わったら`make`でビルドします。

```
make -j 4 all
make test
make runtest
```

Pythonを使う場合は以下のように依存関係をインストールすることができます。

```bash
sudo apt-get install python-dev python-numpy python-pip python-scipy
for req in $(cat python/requirements.txt) pydot; do sudo pip install $req; done
```

Pythonインタフェースをビルドするには`make`のターゲットに`pycaffe`を指定します。

```bash
make pycaffe
```

Caffeとは直接関係ありませんが、Jupyter notebookを使うには`pip`を使います。

```bash
sudo pip install jupyter
```

## ネットワーク構造のグラフ画像

Caffeの配布パッケージには様々なデータやツールが含まれています。その中にネットワークのレイヤー構造をグラフとして画像に出力するスクリプトが含まれています。使うにはGraphvizと`pydotplus` Pythonパッケージをインストールしておく必要があります。

*インストール手順*

ターミナルを開き、以下を入力します。

```bash
apt-get update
apt-get install graphviz
pip install pydotplus
```

*使い方*

コマンドラインから立ち上げて使います。

```bash
cd $CAFFE_ROOT
python python/draw_net.py examples/mnist/lenet.prototxt lenet.png --rankdir=TB
```
最初の引数はネットワークが定義されているprototxtファイルのパス、二番目は出力されるグラフ画像のパス、`--rankdir`オプションは入力から出力がグラフ内でどのような向きになるかを指定するものです。`TB`はTop-to-Bottomの略で、上に入力、下に出力が来るようにグラフを描画します。
