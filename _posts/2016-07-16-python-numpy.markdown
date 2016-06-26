---
layout: post
title:  "1. Pythonと数値計算"
date:   2016-07-16 12:00:00 +0900
category: python
---

* TOC
{:toc}

## はじめに

このページは[Stanford CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/)を参考に作られています。

この実習ではPythonプログラミング言語を使用します。Pythonは汎用的なインタプリタ型言語で、数値計算(numpy, scipy)やプロット表示(matplotlib)など科学技術計算向けのライブラリが充実しているため、近年画像処理用途にも人気が高まっています。

この実習は多少のプログラミング経験のある人を対象にしていますが、Pythonでの数値計算の扱いに不慣れな人のために以下で簡単な使い方を紹介します。これまでにMatlabを使用したことのある人は[numpy for Matlab users](http://wiki.scipy.org/NumPy_for_Matlab_Users)を参考にするもの良いでしょう。日本語資料が必要な人は電通大の庄野先生の[プログラミング言語演習 Python](http://daemon.ice.uec.ac.jp/~shouno/2012.Programming/)などが大いに参考になると思います。言語仕様の詳細は[ドキュメント](http://docs.python.jp/2/)を参考にしてください。

ここでは以下の話題について解説します。

 * Pythonの基本：データ型（コンテナ、リスト、辞書、集合、タプル）、関数、クラス
 * Numpy：配列、配列のインデックス、データ型、演算、ブロードキャスト
 * Matplotlib：プロット、サブプロット、画像表示

## Pythonの基本

Pythonは汎用目的で開発された、動的型付きのインタプリタ型言語です。Pythonを使うと複雑なアルゴリズムも、かなり少ない行数で、読みやすい形で書くことができます。例えばクイックソートアルゴリズムをPythonで書くと以下のようになります。

{% highlight python %}
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) / 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print quicksort([3,6,8,10,1,2,1])
#=> [1, 1, 2, 3, 6, 8, 10]
{% endhighlight %}

### Pythonのバージョン

現在Pythonには2.x系と3.x系があります。Python 3.0以降では文法構造に変更があり、バージョン2.7以前のコードがそのままではバージョン3.0以降では動かないことがあります。例えば`print`関数の文法では以下のようなな違いがあります。この教材ではバージョン2.7を前提にしてコードを記述します。

{% highlight python %}
print 'Python 2.x syntax'   # Python 2.xでは括弧なし
print('Python 3.x syntax')  # Python 3.xでは括弧あり
{% endhighlight %}

### データ型

#### 数値

整数や浮動小数点型は他の言語と同じように使うことができます。

{% highlight python %}
x = 3
print x, type(x)
#=> 3 <type 'int'>
{% endhighlight %}

{% highlight python %}
print x + 1   # 加算
print x - 1   # 減算
print x * 2   # 乗算
print x ** 2  # 冪乗
{% endhighlight %}

{% highlight python %}
x += 1
print x  #=> "4"
x *= 2
print x  #=> "8"
{% endhighlight %}

{% highlight python %}
y = 2.5
print type(y) #=> "<type 'float'>"
print y, y + 1, y * 2, y ** 2 #=> "2.5 3.5 5.0 6.25"
{% endhighlight %}

Pythonにはインクリメント（`x++`）、デクリメント（`x--`）はありません。Pythonには長整数型や複素数型もあります。詳しくは[ドキュメント](http://docs.python.jp/2/library/functions.html)を参照してください。

#### 論理型

Pythonでは論理演算は使えますが、演算子（`&&, ||`）ではなく英単語が演算子になっています。

{% highlight python %}
t = True
f = False
print type(t) #=> "<type 'bool'>"
print t and f # 論理積; "False"
print t or f  # 論理和; "True"
print not t   # 否定; "False"
print t != f  # 排他的論理和; "True"
{% endhighlight %}

#### 文字列

文字列は通常通り使えます。

{% highlight python %}
hello = 'hello'    # シングルクォートか、
world = "world"    # ダブルクォートで文字列を表す
print hello        #=> "hello"
print len(hello)   # 文字列の長さ;=> "5"
hw = hello + ' ' + world  # 文字列の結合
print hw           #=> "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # 文字列の代入
print hw12         #=> "hello world 12"
nihongo = u'日本語' # 明示的にUnicode型を作るにはuをつける
print nihongo      #=> '日本語'
{% endhighlight %}

文字列の操作には便利なメソッドが予め定義されています。

{% highlight python %}
s = "hello"
print s.capitalize()  #=> "Hello"
print s.upper()       #=> "HELLO"
print s.rjust(7)      # 右寄せ;=> "  hello"
print s.center(7)     # 中央寄せ;=> " hello "
print s.replace('l', '(ell)')  # 部分文字列の置換;=> "he(ell)(ell)o"
print '  world '.strip()  # 空白の削除;=> "world"
{% endhighlight %}

[ドキュメント](http://docs.python.jp/2/library/stdtypes.html#string-methods)に全てのメソッドが書かれています。

#### コンテナ型

Pythonには多数のコンテナ型が定義されています。（リスト型、辞書型、集合型、タプル型など）

##### リスト

リストはPythonで言う所の配列に相当します。ただし、リストはサイズを変えることができ、また複数のデータ型を含むことができます。

{% highlight python %}
xs = [3, 1, 2]   # リスト作成
print xs, xs[2]  #=> "[3, 1, 2] 2"
print xs[-1]     # 負のインデックス値は後ろから数えます;=> "2"
xs[2] = 'foo'    # リストには複数のデータ型を混ぜられます
print xs         #=> "[3, 1, 'foo']"
xs.append('bar') # データを後ろに追加します
print xs         #=> "[3, 1, 'foo', 'bar']"
x = xs.pop()     # データを後ろから取り除きます
print x, xs      #=> "bar [3, 1, 'foo']"
{% endhighlight %}

[ドキュメント](http://docs.python.org/2/tutorial/datastructures.html#more-on-lists)に全ての仕様が書いてあります。

**スライス**：リストの要素には一つ一つアクセスするだけでなく、範囲指定により部分リストを抜き出すスライス文法があります。

{% highlight python %}
nums = range(5)    # rangeは整数範囲を作成する組み込み型関数です
print nums         #=> "[0, 1, 2, 3, 4]"
print nums[2:4]    # インデックス2以上4未満;=> "[2, 3]"
print nums[2:]     # インデックス2以上最後まで;=> "[2, 3, 4]"
print nums[:2]     # 最初から2未満;=> "[0, 1]"
print nums[:]      # 全要素;=> ["0, 1, 2, 3, 4]"
print nums[:-1]    # 負のインデックスも可能;=> ["0, 1, 2, 3]"
nums[2:4] = [8, 9] # 範囲に代入も可能
print nums         #=> "[0, 1, 8, 9, 4]"
{% endhighlight %}

スライス文法はnumpy配列を扱うときにまた出てきます。


**ループ**：ループは以下のように行います。

{% highlight python %}
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print animal
#=> "cat", "dog", "monkey"
{% endhighlight %}

配列のインデックスは以下のように`enumerate`関数を使います。

{% highlight python %}
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print '#%d: %s' % (idx + 1, animal)
#=> "#1: cat", "#2: dog", "#3: monkey"
{% endhighlight %}

**リスト内包表記（List comprehension）**：プログラムを書いているときに配列の中身を全て変換したいことがあります。例えば以下のコードをみてください。

{% highlight python %}
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print squares   #=> [0, 1, 4, 9, 16]
{% endhighlight %}

こんな時はブラケットを使って以下のように内包表記で書くことができます。

{% highlight python %}
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print squares   #=> [0, 1, 4, 9, 16]
{% endhighlight %}

リスト内包表記は条件判定を含めることもできます。

{% highlight python %}
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print even_squares  #=> "[0, 4, 16]"
{% endhighlight %}

##### 辞書

辞書型は(key, value)のペアを格納する型で、Javaの`Map`やJavascriptの`Object`と似ています。使い方は以下のようになります。

{% highlight python %}
d = {'cat': 'cute', 'dog': 'furry'}  # 辞書型を作ります
print d['cat']       # 要素にアクセスします;=> "cute"
print 'cat' in d     # 与えられたキーがあるか判定します;=> "True"
d['fish'] = 'wet'    # 上書きします
print d['fish']      #=> "wet"
# print d['monkey']  # KeyError: 'monkey' not a key of d
print d.get('monkey', 'N/A')  # なければデフォルトの値を返します;=> "N/A"
print d.get('fish', 'N/A')    # あればそのキーの値を返します;=> "wet"
del d['fish']        # 要素を取り除きます
print d.get('fish', 'N/A') # "fish"はもうなくなりました;=> "N/A"
{% endhighlight %}

[ドキュメント](http://docs.python.jp/2/library/stdtypes.html#dict)に仕様が書かれています。

**ループ**：ループは以下のように簡単にできます。

{% highlight python %}
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print 'A %s has %d legs' % (animal, legs)
#=> "A person has 2 legs", "A spider has 8 legs", "A cat has 4 legs"
{% endhighlight %}

キーも一緒に使うには`iteritems`メソッドを使います。

{% highlight python %}
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.iteritems():
    print 'A %s has %d legs' % (animal, legs)
#=> "A person has 2 legs", "A spider has 8 legs", "A cat has 4 legs"
{% endhighlight %}

**辞書内包表記**：リスト内包表記に似たようなやり方で辞書を作ることができます。例えば以下のようなことができます。

{% highlight python %}
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print even_num_to_square  #=> "{0: 0, 2: 4, 4: 16}"
{% endhighlight %}

##### 集合

集合は順序を持たず、同一でない要素の集まりです。単純な例を以下に示します。

{% highlight python %}
animals = {'cat', 'dog'}
print 'cat' in animals   # 要素が集合の中にあるか判定します;=> "True"
print 'fish' in animals  #=> "False"
animals.add('fish')      # 新しい要素を加えます
print 'fish' in animals  #=> "True"
print len(animals)       # 集合内の要素数を数えます;=> "3"
animals.add('cat')       # すでに存在する要素を加えても集合は増えません
print len(animals)       #=> "3"
animals.remove('cat')    # 集合から要素を取り除きます
print len(animals)       #=> "2"
{% endhighlight %}

他と同様に[ドキュメント](http://docs.python.jp/2/library/sets.html#set-objects)に仕様が書いてあります。

**ループ**：ループは他と同じような文法でできます。ただし、要素の順番は規定されていないので、どの要素がどの順番で現れるかは実行するまでわかりません。

{% highlight python %}
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print '#%d: %s' % (idx + 1, animal)
#=> "#1: fish", "#2: dog", "#3: cat"
{% endhighlight %}

**集合内包表記**：集合も内包表記ができます。

{% highlight python %}
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print nums  #=> "set([0, 1, 2, 3, 4, 5])"
{% endhighlight %}

##### タプル

タプルは上書き不可の順序付けされた値の組です。タプルはリストに似ていますが、一番の違いは辞書のキーや集合の要素として使うことができるという点です。以下に一例を示します。

{% highlight python %}
d = {(x, x + 1): x for x in range(10)}  # タプルがキーになっている辞書を作ります
t = (5, 6)       # タプルを作ります
print type(t)    #=> "<type 'tuple'>"
print d[t]       #=> "5"
print d[(1, 2)]  #=> "1"
{% endhighlight %}

詳しくは[ドキュメント](http://docs.python.jp/2/tutorial/datastructures.html#tuples-and-sequences)を見てください。

### 関数

Pythonの関数は`def`キーワードで定義されます。以下、一例。

{% highlight python %}
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print sign(x)
#=> "negative", "zero", "positive"
{% endhighlight %}

引数にオプションを使うということがよくありますが、そんな時は以下のようにします。

{% highlight python %}
def hello(name, loud=False):
    if loud:
        print 'HELLO, %s!' % name.upper()
    else:
        print 'Hello, %s' % name

hello('Bob') #=> "Hello, Bob"
hello('Fred', loud=True)  #=> "HELLO, FRED!"
{% endhighlight %}

[ドキュメント](http://docs.python.jp/2/tutorial/controlflow.html#defining-functions)に仕様が書かれています。

### クラス

Pythonのクラスはわかりやすい文法です。

{% highlight python %}
class Greeter(object):

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print 'HELLO, %s!' % self.name.upper()
        else:
            print 'Hello, %s' % self.name

g = Greeter('Fred')  # Greeterクラスのインスタンスを作成
g.greet()            # インスタンスメソッドを呼び出し;=> "Hello, Fred"
g.greet(loud=True)   # インスタンスメソッドを呼び出し;=> "HELLO, FRED!"
{% endhighlight %}

クラスについても[ドキュメント](http://docs.python.jp/2/tutorial/classes.html)を参照してください。

## Numpy

[Numpy](http://www.numpy.org/)はPythonでの科学技術計算の中心となるライブラリです。高性能な多次元配列オブジェクトやそれを扱うツールなどが揃っています。もしMatlabを使ったことがある人は[このチュートリアル](http://wiki.scipy.org/NumPy_for_Matlab_Users)が参考になると思います。

### 配列

Numpyにおける配列は要素値のグリッドで、全てが同一型となっており、非負整数のタプル型でインデックスされています。次元の数は配列の_ランク(rank)_となっています。配列の_形(shape)_は整数のタプルで表現され、次元毎に大きさを示しています。

Numpy配列はネスとされたPythonリストから作成することができ、要素にはブラケットを使ってアクセスできます。

{% highlight python %}
import numpy as np

a = np.array([1, 2, 3])  # ランク1の配列
print type(a)            #=> "<type 'numpy.ndarray'>"
print a.shape            #=> "(3,)"
print a[0], a[1], a[2]   #=> "1 2 3"
a[0] = 5                 # 配列の要素を書き換え
print a                  #=> "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])   # ランク2の配列作成
print b.shape                     #=> "(2, 3)"
print b[0, 0], b[0, 1], b[1, 0]   #=> "1 2 4"
{% endhighlight %}

Numpyではこの他にも多数の配列作成関数が用意されています。

{% highlight python %}
import numpy as np

a = np.zeros((2,2))  # 全て0の配列作成
print a              #=> "[[ 0.  0.]
                     #     [ 0.  0.]]"

b = np.ones((1,2))   # 全て1の配列作成
print b              #=> "[[ 1.  1.]]"

c = np.full((2,2), 7) # 定数配列を作成
print c               #=> "[[ 7.  7.]
                      #     [ 7.  7.]]"

d = np.eye(2)        # 2x2の単位行列を作成
print d              #=> "[[ 1.  0.]
                     #     [ 0.  1.]]"

e = np.random.random((2,2)) # 乱数で行列を作成
print e                     #=> "[[ 0.91940167  0.08143941]
                            #     [ 0.68744134  0.87236687]]"
{% endhighlight %}

[ドキュメント](http://docs.scipy.org/doc/numpy/user/basics.creation.html#arrays-creation)にその他の配列作成方法が記述されています。

### 配列のインデックス

Numpyでは複数のインデックス方法が用意されています。

**スライス**：Pythonリストのように、numpy配列もスライスできます。配列は多次元なので、次元毎にスライスを指定する必要があります。

{% highlight python %}
import numpy as np

# ランク2で形が (3, 4) の配列を作成
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# 最初の2行と、1番と2番の列をスライスします。bの形は (2, 2) になります。
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

# スライスは元の配列を参照したままです。従って、スライスを書き換えると
# 元の配列も書き換わってしまいます。
print a[0, 1]   #=> "2"
b[0, 0] = 77    # b[0, 0] は a[0, 1] と同じ要素を指しています
print a[0, 1]   #=> "77"
{% endhighlight %}

整数値のインデックスとスライスのインデックスは混ぜて使うことができます。しかし、そのようにして抜き出した配列は元の配列に比べてランクが小さくなります。この点はMatlabとは扱いが違うので注意が必要です。

{% highlight python %}
import numpy as np

# ランク2で大きさ (3, 4) の行列を作ります
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# 中間行にアクセスする2つの方法
# 整数インデックスとスライスを混ぜた場合はランクが小さな配列が返されるが、
# スライスのみを使った場合は元と同じランクで配列が返される
row_r1 = a[1, :]    # 配列aの2行目からランク1で抜き出したView
row_r2 = a[1:2, :]  # 配列aの2行目からランク2で抜き出したView
print row_r1, row_r1.shape  #=> "[5 6 7 8] (4,)"
print row_r2, row_r2.shape  #=> "[[5 6 7 8]] (1, 4)"

# 列にアクセスする場合にも同じことが言えます
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print col_r1, col_r1.shape  #=> "[ 2  6 10] (3,)"
print col_r2, col_r2.shape  #=> "[[ 2]
                            #     [ 6]
                            #     [10]] (3, 1)"
{% endhighlight %}

**整数インデックス**：スライスでNumpy配列にアクセスした場合、返される配列のViewは常に元の配列の部分配列になります。しかし、整数でインデックスをした場合は任意の配列を他の配列から作ることができます。以下に例を示します。

{% highlight python %}
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

# 整数インデックスの例

# 返ってくる配列の形は (3,)
print a[[0, 1, 2], [0, 1, 0]]  #=> "[1 4 5]"

# 上記の整数インデックスは以下と同等
print np.array([a[0, 0], a[1, 1], a[2, 0]])  #=> "[1 4 5]"

# 整数インデックスを使った場合は元の配列の同じ要素を再利用できる
print a[[0, 0], [1, 1]]  #=> "[2 2]"

# 上記の整数インデックスは以下と同等
print np.array([a[0, 1], a[0, 1]])  #=> "[2 2]"
{% endhighlight %}

整数インデックスの便利な使い道としては、行列の各行から一つずつ要素を選択したり上書きしたりする用途がある。

{% highlight python %}
import numpy as np

# インデックスに使う新しい配列を作成
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print a  #=> "array([[ 1,  2,  3],
         #           [ 4,  5,  6],
         #           [ 7,  8,  9],
         #           [10, 11, 12]])"

# インデックスの配列を作成
b = np.array([0, 2, 0, 1])

# インデックス配列 b を使って各行から一つずつ要素を抜き出し
print a[np.arange(4), b]  #=> "[ 1  6  7 11]"

# インデックス配列 b を使って各要素を上書き
a[np.arange(4), b] += 10

print a  #=> "array([[11,  2,  3],
         #           [ 4,  5, 16],
         #           [17,  8,  9],
         #           [10, 21, 12]])
{% endhighlight %}

**論理インデックス**：論理インデックスを使って配列から任意の要素を指定することができます。この方法はよく条件判定と組み合わせて使われます。以下に一例を示します。

{% highlight python %}
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)  # 2より大きな要素を探します
                    # これで配列 a の要素のうち、要素の位置が2よりも大きな値を持つか
                    # どうかを判定する論理値を持つ、元の配列ど同一の大きさの論理配列が
                    # bool_idxに格納されます

print bool_idx      #=> "[[False False]
                    #     [ True  True]
                    #     [ True  True]]"

# 論理インデックスを使ってランク1でbool_idxがTrueのところのみからなる配列を作ります
print a[bool_idx]  #=> "[3 4 5 6]"

# 上記の例は1行で書くことができます
print a[a > 2]     #=> "[3 4 5 6]"
{% endhighlight %}

Numpyの配列インデックスについてはかなり省略して紹介しています。詳しくは[ドキュメント](http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)をよく読んでください。

### データ型

Numpyの配列要素は必ず同一の型です。Numpyでは多数の数値型を使って配列を作ることができます。Numpyは配列の作成時に自動でデータ型を選択します。しかし、配列の作成時にオプションでデータ型を指定することができます。以下に例を示します。

{% highlight python %}
import numpy as np

x = np.array([1, 2])  # 自動でデータ型を選択させます
print x.dtype         #=> "int64"

x = np.array([1.0, 2.0])  # 自動でデータ型を選択させます
print x.dtype             #=> "float64"

x = np.array([1, 2], dtype=np.int64)  # 特定のデータ型を指定します
print x.dtype                         #=> "int64"
{% endhighlight %}

詳細は[ドキュメント](http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html)を参照してください。

### 配列の演算

基本的な数学関数は配列の全ての要素について計算します。演算はオーバーロードされたpython演算子としてもnumpyモジュールの関数としても利用できます。

{% highlight python %}
import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# 要素加算; どちらも配列を生成
# [[ 6.0  8.0]
#  [10.0 12.0]]
print x + y
print np.add(x, y)

# 要素減算; どちらも配列を生成
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print x - y
print np.subtract(x, y)

# 要素乗算; どちらも配列を生成
# [[ 5.0 12.0]
#  [21.0 32.0]]
print x * y
print np.multiply(x, y)

# 要素除算; どちらも配列を生成
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print x / y
print np.divide(x, y)

# 要素平方根; どちらも配列を生成
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print np.sqrt(x)
{% endhighlight %}

Matlabとは異なり、`*`演算子は行列積ではなく要素積となっていることに気をつけてください。ベクトルの内積や行列積には`dot`関数を使います。`dot`はnumpyモジュールの関数か、配列のメソッドとして利用可能です。

{% highlight python %}
import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# ベクトルの内積、どちらも219
print v.dot(w)
print np.dot(v, w)

# 行列とベクトルの積、どちらもランク1の配列 [29 67] を生成
print x.dot(v)
print np.dot(x, v)

# 行列積、rank 2 arrayを生成
# [[19 22]
#  [43 50]]
print x.dot(y)
print np.dot(x, y)
{% endhighlight %}

Numpyには配列での演算をする多数の関数が用意されています。最も使う機会が多いのが`sum`でしょう。

{% highlight python %}
import numpy as np

x = np.array([[1,2],[3,4]])

print np.sum(x)  # 全ての要素の和;=> "10"
print np.sum(x, axis=0)  # 列ごとに和;=> "[4 6]"
print np.sum(x, axis=1)  # 行ごとに和;=> "[3 7]"
{% endhighlight %}

他の数学関数は[Numpyドキュメント](http://docs.scipy.org/doc/numpy/reference/routines.math.html)に載っています。

配列の数学演算の他に、配列の形を変えたり値を操作したりする必要もよくあります。一番わかりやすい例が行列の転置でしょう。行列の転置には`T`アトリビュートを使います。

{% highlight python %}
import numpy as np

x = np.array([[1,2], [3,4]])
print x    #=> "[[1 2]
           #     [3 4]]"
print x.T  #=> "[[1 3]
           #     [2 4]]"

# ランク1配列の転置には意味がありません
v = np.array([1,2,3])
print v    #=> "[1 2 3]"
print v.T  #=> "[1 2 3]"
{% endhighlight %}

この他の配列操作方法は[ドキュメント](http://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html)を読みましょう。

### ブロードキャスト

ブロードキャストは大きさの異なる配列を扱うときに力を発揮する仕組みです。よくあるケースが、小さな配列と大きな配列があって、小さな配列を何度も大きな配列に対して演算したいというものです。例えば、行列の各行に定数のベクトルを足し合わせたいという場合を見てみましょう。

{% highlight python %}
import numpy as np

# 行列 x の各行にベクトル v を足し合わせたいケースを考えます
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # x と同じ大きさの空白の行列を作ります

# ループを使って各行にベクトル v を足し合わせます
for i in range(4):
    y[i, :] = x[i, :] + v

# y の中身は以下のようになります
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print y
{% endhighlight %}

この方法は正しく動きます。ただし、行列`x`が非常に大きい場合には、Pythonでループを使って計算するとかなり遅くなってしまいます。行列`x`の各行にベクトル`v`を足し合わせるという演算は、ベクトル`v`のコピーを縦に積み重ねて行列`vv`を作り、それから`x`と`vv`の要素和を取ることと等価です。これを以下のようにやることができます。

{% highlight python %}
import numpy as np

# 行列 x の各行にベクトル v を足し合わせたいケースを考えます
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))  # v のコピーを積み重ねます
print vv                 #=> "[[1 0 1]
                         #     [1 0 1]
                         #     [1 0 1]
                         #     [1 0 1]]"
y = x + vv  # x と vv を足します
print y  #=> "[[ 2  2  4
         #     [ 5  5  7]
         #     [ 8  8 10]
         #     [11 11 13]]"
{% endhighlight %}

Numpyのブロードキャストを用いると実際に`v`を複数回コピーすることなく同じことができます。以下のブロードキャストの使用例を見てください。

{% highlight python %}
import numpy as np

# 行列 x の各行にベクトル v を足し合わせたいケースを考えます
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # v をブロードキャストを使って x に足します
print y  #=> "[[ 2  2  4]
         #     [ 5  5  7]
         #     [ 8  8 10]
         #     [11 11 13]]"
{% endhighlight %}

`y = x + v`の行は`x`が`(4, 3)`の大きさで`v`が`(3,)`の大きさにも関わらず、ブロードキャストのおかげで計算ができます。ブロードキャストをすると`v`の大きさがあたかも`(4, 3)`で各行に`v`のコピーが存在するかのように取り扱われるためです。

ブロードキャストは以下のようなルールで使われます。

  1. 二つの配列が同じランクでない場合、小さいランクの配列の形にどちらの配列の形状が同じになるまで1を付け加えます
  2. もし二つの配列が同一の次元を持っているか、片方の配列がサイズ1の次元を持っているとき、二つの配列は次元に関して_互換性_があるといいます
  3. 配列に互換性があるとき、ブロードキャストが可能です
  4. ブロードキャストすると、双方の配列は二つの配列の形状のうち大きい方の形を持っているものとして取り扱われます
  5. 大きさ1の次元を持つ配列があり、もう片方の配列の次元の大きさが1より大きい場合、最初の配列はその次元に沿ってコピーされたように振舞います

この説明でよくわからない人は[ドキュメント](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)や[この説明](http://wiki.scipy.org/EricsBroadcastingDoc)を見てください。

関数のうちブロードキャストをサポートするものは_ユニバーサル関数_と呼ばれます。ユニバーサル関数のリストは[ドキュメント](http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs)に書かれています。

以下にブロードキャストの使い方の例を示します。

{% highlight python %}
import numpy as np

# ベクトル同士の積
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# 外積を計算にはまず v を (3, 1) の列ベクトルにし、続いて w にブロードキャストして
# (3, 2) の行列を作ります
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print np.reshape(v, (3, 1)) * w

# ベクトルを行列の行に足す
x = np.array([[1,2,3], [4,5,6]])
# x は (2, 3) の大きさで v は (3,) なのでブロードキャストで (2, 3) になる
# [[2 4 6]
#  [5 7 9]]
print x + v

# ベクトルを行列の列に足す
# x は (2, 3) で w は (2,)
# x の転置を作ると (3, 2) となり、 w にブロードキャストすると (3, 2) になる
# これを更に転置すると (2, 3) となって、元の x の各列に w を足した結果となる
# [[ 5  6  7]
#  [ 9 10 11]]
print (x.T + w).T
# あるいは、w の形状を変更して (2, 1) にすると直接ブロードキャストできる
print x + np.reshape(w, (2, 1))

# 行列を定数倍
# x は (2, 3) の大きさ。Numpyではスカラは大きさ () の配列として扱われる。
# これらはブロードキャストして (2, 3) の配列として使うことができる
# [[ 2  4  6]
#  [ 8 10 12]]
print x * 2
{% endhighlight %}

ブロードキャストは一般にコードを簡潔にし、しかも速く実行できるようにするので、使える場面ではなるべく使うようにしましょう。

### Numpyドキュメント

以上がNumpyについての重要な使い方の説明ですが、まだまだ多くのことがあります。詳細はNumpyの[ドキュメント](http://docs.scipy.org/doc/numpy/reference/)に目を通すようにしましょう。


## SciPy

Numpyは高性能な配列とそれを扱う関数を揃えています。SciPyはNumpyをベースに作られたライブラリで、Numpyの配列を扱うための様々な科学技術計算に用いる関数を揃えています。

SciPyを知るための一番の方法は[ドキュメント](http://docs.scipy.org/doc/scipy/reference/index.html)を読むことです。ここでは以下にSciPyのうち実習に関連がありそうなものを紹介します。

### 画像の操作

{% highlight python %}
from scipy.misc import imread, imsave, imresize

# JPEGファイルをNumpy配列に読み込み
img = imread('assets/cat.jpg')
print img.dtype, img.shape  #=> "uint8 (400, 248, 3)"

# 配列を操作して画像の色合いを少し変えることができます。画像は (400, 248, 3) の
# 大きさです。ここに [1, 0.95, 0.9] を掛け合わせます
# Numpyブロードキャストを使うと、赤色はそのまま、緑色は0.95倍、青色は0.9倍になって
# 少し色あせたような画像ができます
img_tinted = img * [1, 0.95, 0.9]

# 画像を 300 x 300 ピクセルにリサイズします
img_tinted = imresize(img_tinted, (300, 300))

# 画像をディスクに保存します
imsave('assets/cat_tinted.jpg', img_tinted)
{% endhighlight %}

![cat]({{ site.github.url }}/assets/cat.jpg)
![cat_tinted]({{ site.github.url }}/assets/cat_tinted.jpg)
{: style="text-align: center;"}

Left: The original image. Right: The tinted and resized image.

### MATLABファイル

Matlab形式のファイルは`scipy.io.loadmat`関数と`scipy.io.savemat`関数で読み書きすることができます。詳しくは[ドキュメント](http://docs.scipy.org/doc/scipy/reference/io.html)を参照。

### 点と点の距離

SciPyには点群の距離を計算する便利な関数が含まれています。

`scipy.spatial.distance.pdist`関数は全ての点同士の距離を計算する関数です。

{% highlight python %}
import numpy as np
from scipy.spatial.distance import pdist, squareform

# 各行に2次元の点の座標が格納された行列
# [[0 1]
#  [1 0]
#  [2 0]]
x = np.array([[0, 1], [1, 0], [2, 0]])
print x

# 全ての行同士のユークリッド距離を計算
# d[i, j] は x[i, :] と x[j, :] の間の距離を示す
# dの中身は以下の通り
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
d = squareform(pdist(x, 'euclidean'))
print d
{% endhighlight %}

[ドキュメント](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html)に詳細が載っています。

似たような関数に`scipy.spatial.distance.cdist`があります。これは2つの点群間の全てのペアから距離を計算します。詳しくは[ドキュメント](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)。

## Matplotlib

[Matplotlib](http://matplotlib.org/)はプロット用のライブラリです。ここでは`matplotlib.pyplot`モジュールを簡単に紹介します。これはMatlabに似たプロット機能を提供するライブラリです。

### プロット

一番重要な関数は`plot`です。これは2Dデータのプロットを行います。以下に例を示します。

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt

# 正弦波のxy座標を計算
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Matplotlibでプロット表示
plt.plot(x, y)
plt.show()  # 最後に plt.show() を呼ばないと表示されないので注意
{% endhighlight %}

実際に動かすと以下のようなプロットが表示されます。

![sine]({{ site.github.url }}/assets/sine.png)
{: style="text-align: center;"}

もう少し付け足すと複数のプロットを一度に行ったりタイトル、凡例、軸のラベルなどをつけることができます。

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt

# 正弦波のxy座標を計算
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Matplotlibでプロット表示
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
{% endhighlight %}

![sine_cosine]({{ site.github.url }}/assets/sine_cosine.png)
{: style="text-align: center;"}

`plot`の細かい使い方は[ドキュメント](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot)を参照。

### サブプロット

`subplot`を使うと複数のプロットは同じ図に表示することができます。以下に例を示します。

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt

# 正弦波のxy座標を計算
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# 高さ2幅1のサブプロットを設定して最初のプロットを指定
plt.subplot(2, 1, 1)

# 最初のプロットを描画
plt.plot(x, y_sin)
plt.title('Sine')

# 二番目のプロットを指定して描画
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# 図を表示
plt.show()
{% endhighlight %}

![sine_cosine_subplot]({{ site.github.url }}/assets/sine_cosine_subplot.png)
{: style="text-align: center;"}

詳しくは`subplot`の[ドキュメント](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot)を参照。

### 画像表示

画像を表示するには`imshow`を使います。

{% highlight python %}
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.95, 0.9]

# 元の画像を表示
plt.subplot(1, 2, 1)
plt.imshow(img)

# 色あせた画像を表示
plt.subplot(1, 2, 2)

# imshowはuint8でないデータ型をを使うと変な表示になることがあります
# これを防ぐには画像を明示的にuint8型にキャストしましょう
plt.imshow(np.uint8(img_tinted))
plt.show()
{% endhighlight %}

![cat_tinted_imshow]({{ site.github.url }}/assets/cat_tinted_imshow.png)
{: style="text-align: center;"}

