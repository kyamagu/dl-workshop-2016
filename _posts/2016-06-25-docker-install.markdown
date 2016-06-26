---
layout: post
title:  "0. Docker+Jupyter環境の構築"
date:   2016-06-25 12:59:05 +0900
category: python
---

 * TOC
{:toc}

## Dockerとは

[Docker](https://www.docker.com/)はコンテナ化による仮想実行環境を構築するソフトウェアです。VMWareなどの仮想マシンに似ていますが、OSを丸ごと仮想化するのではなくコンテナと呼ばれる実行単位で仮想環境を構築し、軽量かつ高速な点が特徴となっています。また、[Docker hub](https://hub.docker.com/)サービスを通してコンテナイメージの配布も簡易に行うことができます。

CaffeフレームワークはWindows環境で動作させることは前提としていないため、この実習ではDockerの上にUbuntu Linux環境を構築してCaffeフレームワークを利用します。また、ユーザーがDockerコンテナを操作するためのインタフェースとしてPythonのJupyter notebook環境を利用します。

## Dockerのインストール

最初にOS毎にDockerをインストールする手順を紹介します。

### Windows

Dockerは2016年6月以前はToolbox版によるインストールが主流でしたが、現在はDocker for Windowsアプリケーションに置き換わっています。Docker for WindowsはWindows 10 64-bit環境で利用可能です。インストーラは以下のURLから入手できます。

[https://www.docker.com/products/docker#/windows](https://www.docker.com/products/docker#/windows)

PCによってはBIOSの設定でCPUの仮想マシン支援機能を有効にする必要があるかもしれません。また、メモリが少ないPCではDockerの起動に失敗することがあります。この時はDockerの設定画面から起動時の使用メモリを1024MB程度まで下げてから起動を行ってください。

### Mac

Dockerは2016年6月以前はToolbox版によるインストールが主流でしたが、現在はDocker for Windowsアプリケーションに置き換わっています。インストーラは以下のURLから入手できます。

[https://www.docker.com/products/docker#/mac](https://www.docker.com/products/docker#/mac)

なお、MacではDockerを使わなくてもCaffeをコンパイルして使うことができます。ただしMacには高性能GPUの入ったワークステーションがないため実用には不向きです。

### Ubuntu Linux

UbuntuではDocker公式のAPTレポジトリを追加することで`apt-get`経由でインストールできるようになります。詳細な手順は以下のURLを参考にしてください。

[https://www.docker.com/products/docker#/linux](https://www.docker.com/products/docker#/linux)

Ubuntu 16.04 LTSでは以下の手順でインストールできます。

{% highlight bash %}
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates
sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
sudo tee /etc/apt/sources.list.d/docker.list <<EOF
deb https://apt.dockerproject.org/repo ubuntu-xenial main
EOF
sudo apt-get update
sudo apt-get install linux-image-extra-$(uname -r)
sudo apt-get install docker
{% endhighlight %}

## 実習用Dockerイメージのダウンロードと起動

Dockerをインストールした後は実習で利用するコンテナのイメージをダウンロードして起動します。コンソールやターミナルを起動し、以下のコマンドを入力してください。イメージは2GB程度あるのでダウンロードに多少時間がかかります。

{% highlight bash %}
docker pull kyamagu/caffe
{% endhighlight %}

イメージを起動するには以下のようにコマンドを入力してください。

{% highlight bash %}
docker run -it -p 8888:8888 kyamagu/caffe
{% endhighlight %}

これでDockerコンテナが起動した状態になります。Webブラウザを開き、[http://localhost:8888/](http://localhost:8888/)にアクセスしてみてください。Jupyter notebookの画面が表示されると思います。

![Jupyter notebook](/assets/jupyter-screen1.png)
{: style="text-align:center;"}

起動しているDockerコンテナを止めるにはコンソールで`Ctrl+C`を2回押すか、`Ctrl+C`の後に`y`を押してエンターキーで終了してください。

### GPUについて

この実習はGPUのないPC環境を想定して作られています。実際に大規模なデータで学習を行う場合にはCPUのみでは計算量的に対処困難なため、十分なメモリのあるGPUを使うことを推奨します。この実習で使うDockerイメージはGPUのない状態を前提にして作られていますが、GPUをサポートした形でDocker環境を構築することも可能です。詳しいことは[NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)と[Caffe Dockerfile](https://github.com/BVLC/caffe/tree/master/docker)の作り方を参考にしてください。

### Dockerコンテナの操作

起動中のコンテナを確認

{% highlight bash %}
docker ps
{% endhighlight %}

起動中のコンテナを停止

{% highlight bash %}
docker stop <Container-ID>
{% endhighlight %}

コンテナを再起動

{% highlight bash %}
docker restart <Container-ID>
{% endhighlight %}

ホストに存在するコンテナの一覧

{% highlight bash %}
docker ps -a
{% endhighlight %}

コンテナを削除

{% highlight bash %}
docker rm <Container-ID>
{% endhighlight %}

全てのコンテナを削除 (UNIX環境のみ)

{% highlight bash %}
docker rm `docker ps --no-trunc -aq`
{% endhighlight %}

[Dockerドキュメント](https://docs.docker.com/)に詳細な使い方が書かれています。また、WindowsとMacユーザーは[Kitematic](https://kitematic.com)というGUIアプリケーションを使ってDockerを操作することもできます。

## Jupyter notebookについて

この実習では[Jupyter notebook](http://jupyter.org/)環境でPythonを扱います。Jupyter notebookはブラウザからPythonをノートブック形式で使用したり、通常のUNIXターミナルを呼び出したり、あるいはPythonのコードを作成するエディタを使用できる環境です。この実習ではDockerイメージを起動した時に自動でJupyter notebookを動かすように設定されています。

起動時に以下のような画面が表示されると思います。これは現在のディレクトリの一覧を表示しています。実習では`caffe`のコードが入っているディレクトリが置かれた状態になっていると思います。

![Jupyter notebook](/assets/jupyter-screen1.png)
{: style="text-align:center;"}

### Notebookの作成

まずはNotebook形式でPythonプログラムを実行してみましょう。右上の`New`からNotebooks Python 2を選んで新しいノートを作成します。ノートは高機能なPythonコンソールのようなもので、Pythonコードを記入して実行したり、ドキュメントを記述することができます。以下に一例を示します。

![Jupyter notebook](/assets/jupyter-screen2.png)
{: style="text-align:center;"}

ノートブックを使うと様々な操作ができます。また、[ショートカットキーも充実している](http://qiita.com/angelapy/items/998e99b2d0dc991c99f7)ので慣れるとほぼマウスを使わずに全ての操作ができるようになります。このノートでの作業を終了するときはFileからClose and Haltを選びます。

### UNIXターミナル

最初の画面で右上の`New`からTermninalを選ぶとこのコンテナのUNIXコンソールを開くことができます。以下に例を示します。デフォルトでは`sh`を起動するので`bash`を使うときは後から`bash`を起動します。

![Jupyter notebook](/assets/jupyter-screen3.png)
{: style="text-align:center;"}

また、Python notebookは[IPython環境](https://ipython.org/)なので、`!`をつけてノートブック上から直接UNIXコマンドを呼び出すこともできます。

### ファイル編集

この他にも`New`からディレクトリを作ったりテキストファイルを作成することができます。例えばテキストファイルを作ると以下のようにエディタが使えます。これでPythonプログラムを作成することもできます。

![Jupyter notebook](/assets/jupyter-screen4.png)
{: style="text-align:center;"}
