### BPE，wordPiece，ULM，sentencePiece介绍

#### 什么是token

在自然语言处理任务中，`token`是指机器能够理解和处理的最小单位。对于生活中的自然语言，为了让机器能够理解和学习其中的语义，因此就需要将句子转化成机器能够理解的形式，而转化成机器能够处理的最小单元就是`token`。

而将生活中文本语言转化成`token`的过程就是`tokenization`。这个过程也称之为分词过程。而根据不同的分词器可以将句子转化成不同的`token`。

对于***我爱中国***，可以根据不同的分词器，将其划分成***[’我’, ‘爱‘, ’中’, ‘国‘]***和***[’我‘, ’爱‘, ’中国‘]***。

从中可以知道，根据不同的分词器，分词之后的`token`数量也是不一样的。而具体应该怎么划分对训练较好就是本文接下来将要介绍的。

#### 1、tokenizer划分粒度

tokenizer一般有三种粒度，分别为char, word, subword。

**char,字符型:** 就是单个字符，例如a,b,c,d。

**word,单词型:** n-gram, 多个单词等，例如汉语的jieba工具。

**subword,子词型:** 子词单元，代表形式是BPE, Byte-Pair Encoding 字节对编码，即把一个词可以拆分成多个部分，例如单词This被拆分成,[Th] [#is], 例如工具subword-nmt。

**优缺点：**

**char**：字符是一种语言最基本的组成单元，如英文中的a,b,c或中文中的”你，我，他"等。使用字符有如下问题:1.字符数量是有限的通常数量较少，这样在学习每个字符的embedding向量时，每个字符中包含非常多的语义，学习起来比较困难（语义损失）；2.以字符分，会造成序列长度过长，对后续应用造成较大限制。

**word**：词是最自然的语言单元，对于英文来说其天然存在空格进行，切分相对容易，常用的分词器有spaCy和Moses。中文不具备这样的分割符，所以相对困难一些，不过目前也有Jieba、HanlP、LTP等分词器，这些分词器基于规则与模型，可以取得良好的分词效果。使用词时会有2个问题:1.词表通常是基于语料进行分词获得，但遇到新的语料时可能会出现`OOV`（Out Of Vocabulary）的情况；2.词表过于庞大，对于模型来说大部分参数都集中在输入输出层，不利于模型学习，且容易爆内存(显存)。通常情况下词表大小不超过5万。

**subword**：它介于char和word之间，可以很好的平衡词汇量和语义独立性。它的切分准则是常用的词不被切分，而不常见的词切分为子词。

#### 2、BPE(Byte Pair Encoding)

字节对编码主要的目的是为了数据压缩，通过不断的循环迭代，两两配对，每次将频率最高的配对子词加入到词表当中，直到达到此表的大小或者是最高词频为1。

BPE获得Subword的步骤如下：

> 1、 准备足够大的训练语料，并确定期望的Subword词表大小；
> 2、将单词拆分为成最小单元。比如英文中26个字母加上各种符号，这些作为初始词表；
> 3、在语料上统计单词内相邻单元对的频数，选取频数最高的单元对合并成新的Subword单元；
> 4、重复第3步直到达到第1步设定的Subword词表大小或下一个最高频数为1.

每次合并后词表大小可能出现3种变化：

- +1，表明加入合并后的新子词，同时原来的2个子词还保留（2个字词分开出现在语料中）。
- +0，表明加入合并后的新子词，同时原来的2个子词中一个保留，一个被消解（一个子词完全随着另一个子词的出现而紧跟着出现）。
- -1，表明加入合并后的新子词，同时原来的2个子词都被消解（2个字词同时连续出现）。

实际上，随着合并的次数增加，词表大小通常先增加后减小。

得到Subword词表之后，需要对输入模型的句子中的单词进行编码，编码流程如下：

> 1、将词典中的所有子词按照长度由大到小进行排序；
> 2、对于单词w，依次遍历排好序的词典。查看当前子词是否是该单词的子字符串，如果是，则输出当前子词，并对剩余单词字符串继续匹配。
> 3、如果遍历完字典后，仍然有子字符串没有匹配，则将剩余字符串替换为特殊符号输出，如”<unk>”。
> 4、单词的表示即为上述所有输出子词。

解码：如果相邻子词间没有中止符</w>，则将两子词直接拼接，否则两子词之间添加分隔符。

**例子1**

以一个语料库中的一个句子为例：***"FloydHub is the fastest way to build, train and deploy deep learning models. Build deep learning models in the cloud. Train deep learning models."***

1. 拆分，加后缀，统计词频：

|         WORD         | FREQUENCY |       WORD       | FREQUENCY |
| :------------------: | :-------: | :--------------: | :-------: |
|     d e e p      |     3     |  b u i l d   |     1     |
| l e a r n i n g  |     3     |  t r a i n   |     1     |
|      t h e       |     2     |    a n d     |     1     |
|   m o d e l s    |     2     | d e p l o y  |     1     |
| F l o y d h u b  |     1     |  B u i l d   |     1     |
|       i s        |     1     | m o d e l s  |     1     |
|  f a s t e s t   |     1     |     i n      |     1     |
|      w a y       |     1     |  c l o u d   |     1     |
|       t o        |     1     |  T r a i n   |     1     |

2. 建立词表，统计字符频率：

| NUMBER | TOKEN | FREQUENCY | NUMBER | TOKEN | FREQUENCY |
| :----: | :---: | :-------: | :----: | :---: | :-------: |
|   1    |   |    24     |   15   |   g   |     3     |
|   2    |   e   |    16     |   16   |   m   |     3     |
|   3    |   d   |    12     |   17   |   .   |     3     |
|   4    |   l   |    11     |   18   |   b   |     2     |
|   5    |   n   |    10     |   19   |   h   |     2     |
|   6    |   i   |     9     |   20   |   F   |     1     |
|   7    |   a   |     8     |   21   |   H   |     1     |
|   8    |   o   |     7     |   22   |   f   |     1     |
|   9    |   s   |     6     |   23   |   w   |     1     |
|   10   |   t   |     6     |   24   |   ,   |     1     |
|   11   |   r   |     5     |   25   |   B   |     1     |
|   12   |   u   |     4     |   26   |   c   |     1     |
|   13   |   p   |     4     |   27   |   T   |     1     |
|   14   |   y   |     3     |        |       |           |

3. 之后通过第一次迭代，统计两两子词合并的词频，可知，`'e'`和`'d'`合并的`'de'`出现的次数最多，总共出现了7次，如下所示：

|         WORD         | FREQUENCY |        WORD         | FREQUENCY |
| :------------------: | :-------: | :-----------------: | :-------: |
|   **de** e p     |     3     |   b u i l d     |     1     |
| l e a r n i n g  |     3     |   t r a i n     |     1     |
|      t h e       |     2     |     a n d       |     1     |
| m o **de** l s   |     2     | **de** p l o y  |     1     |
| F l o y d h u b  |     1     |   B u i l d     |     1     |
|       i s        |     1     | m o **de** l s  |     1     |
|  f a s t e s t   |     1     |      i n        |     1     |
|      w a y       |     1     |   c l o u d     |     1     |
|       t o        |     1     |   T r a i n     |     1     |

上面加粗的地方表示`'de'`出现的部分，为3+2+1+1=7。

因此，更新的词表为：

|  TOKEN  |  FREQUENCY   |  TOKEN   | FREQUENCY |
| :-----: | :----------: | :------: | :-------: |
|     |      24      |    g     |     3     |
| ***e*** | ***16-7=9*** |    m     |     3     |
| ***d*** | ***12-7=5*** |    .     |     3     |
|    l    |      11      |    b     |     2     |
|    n    |      10      |    h     |     2     |
|    i    |      9       |    F     |     1     |
|    a    |      8       |    H     |     1     |
|    o    |      7       |    f     |     1     |
|    s    |      6       |    w     |     1     |
|    t    |      6       |    ,     |     1     |
|    r    |      5       |    B     |     1     |
|    u    |      4       |    c     |     1     |
|    p    |      4       |    T     |     1     |
|    y    |      3       | ***de*** |  ***7***  |

其中变化的部分使用斜粗体标出。

后续继续不断地迭代，知道达到所需要的标准（预设大小的词表或者是合并后词频最高的词也为1）停止迭代。

**例子2**

假设现在有四个词：***low, lower, newest, widest***。它们在文本中出现的次数分别为4，6，3，5次。

~~~python
语料：{'l o w </w>':4, 'l o w e r </w>':6, 'n e w e s t </w>':3, 'w i d e s t </w>':5}
~~~

初始词表，词表长度为11

~~~python
词表：{'l','o','w','e','r','n','s','t','i','d','</w>'}
~~~

首先计算两两字符合并的词频，可以知道`'lo'`出现的次数最多，为10次，因此，合并`'l'`和`'o'`：

~~~python
语料：{'lo w </w>':4, 'lo w e r </w>':6, 'n e w e s t </w>':3, 'w i d e s t </w>':5}
~~~

合并'l'和'o'之后，因为合并后的语料中不存在`'l'`和`'o'`，因此删除此表中的`'l'`和`'o'`，并添加`'lo'`，词表长度为10，词表的长度变短，对应着上面的`-1`变化

~~~python
词表：{'lo','w','e','r','n','s','t','i','d','</w>'}
~~~

接着继续两两合并，可知`'low'`出现的次数最多，为10次，因此合并：

~~~python
语料：{'low </w>':4, 'low e r </w>':6, 'n e w e s t </w>':3, 'w i d e s t </w>':5}
~~~

合并之后，因为预料中`'lo'`不存在，因此删掉，而`'w'`仍然存在，因此保留，词表长度为10，词表的长度不变，对应着上面的`+0`变化

~~~python
词表：{'low','w','e','r','n','s','t','i','d','</w>'}
~~~

接着继续计算可知，`'es'`出现出现的频率最高，为8次，合并：

~~~python
语料：{'low </w>':4, 'low e r </w>':6, 'n e w es t </w>':3, 'w i d es t </w>':5}
~~~

合并之后`'e'`继续存在，而`'s'`在语料中已经不存在了，因此，删除`'s'`，同时添加`'es'`，词表长度为10，词表的长度不变，对应着上面的`+0`变化

~~~python
词表：{'low','w','e','r','n','s','t','i','d','es','</w>'}
~~~

......

反复调用直到达到预设的subword词表大小或下一个最高频的字节对出现频率为1。

**代码实现：**

~~~python
#BPE实现代码，也可以使用subword-nmt包直接使用
import re, collections

def get_vocab(filename):
    vocab = collections.defaultdict(int)
    with open(filename, 'r', encoding='utf-8') as fhand:
        for line in fhand:
            words = line.strip().split()
            for word in words:
                vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_tokens(vocab):
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens
~~~

对上面的例子进行预处理：

~~~python
vocab = {'l o w </w>': 4, 'l o w e r </w>': 6, 'n e w e s t </w>': 3, 'w i d e s t </w>': 5}
print('==========')
print('Tokens Before BPE')
tokens = get_tokens(vocab)
print('Tokens: {}'.format(tokens))
print('Number of tokens: {}'.format(len(tokens)))
print('==========')

num_merges = 5
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print('Iter: {}'.format(i))
    print('Best pair: {}'.format(best))
    tokens = get_tokens(vocab)
    print('Tokens: {}'.format(tokens))
    print('Number of tokens: {}'.format(len(tokens)))
~~~

结果

~~~python
==========
Tokens Before BPE
Tokens: defaultdict(<class 'int'>, {'l': 10, 'o': 10, 'w': 18, '</w>': 18, 'e': 17, 'r': 6, 'n': 3, 's': 8, 't': 8, 'i': 5, 'd': 5})
Number of tokens: 11
==========
Iter: 1
Best pair: ('l', 'o')
Tokens: defaultdict(<class 'int'>, {'lo': 10, 'w': 18, '</w>': 18, 'e': 17, 'r': 6, 'n': 3, 's': 8, 't': 8, 'i': 5, 'd': 5})
Number of tokens: 10
Iter: 2
Best pair: ('lo', 'w')
Tokens: defaultdict(<class 'int'>, {'low': 10, '</w>': 18, 'e': 17, 'r': 6, 'n': 3, 'w': 8, 's': 8, 't': 8, 'i': 5, 'd': 5})
Number of tokens: 10
Iter: 3
Best pair: ('e', 's')
Tokens: defaultdict(<class 'int'>, {'low': 10, '</w>': 18, 'e': 9, 'r': 6, 'n': 3, 'w': 8, 'es': 8, 't': 8, 'i': 5, 'd': 5})
Number of tokens: 10
Iter: 4
Best pair: ('es', 't')
Tokens: defaultdict(<class 'int'>, {'low': 10, '</w>': 18, 'e': 9, 'r': 6, 'n': 3, 'w': 8, 'est': 8, 'i': 5, 'd': 5})
Number of tokens: 9
Iter: 5
Best pair: ('est', '</w>')
Tokens: defaultdict(<class 'int'>, {'low': 10, '</w>': 10, 'e': 9, 'r': 6, 'n': 3, 'w': 8, 'est</w>': 8, 'i': 5, 'd': 5})
Number of tokens: 9
~~~

**编码与解码**

上面的过程称为编码。解码过程比较简单，如果相邻子词间没有中止符，则将两子词直接拼接，否则两子词之间添加分隔符。 如果仍然有子字符串没被替换但所有 token 都已迭代完毕，则将剩余的子词替换为特殊 token，如 `<unk>`。例如：

~~~python
# 编码序列
["the</w>", "high", "est</w>", "moun", "tain</w>"]

# 解码序列
"the</w> highest</w> mountain</w>"
~~~

#### 3、wordPiece

WordPiece算法可以看作是BPE的变种。不同点在于，WordPiece基于**概率**生成新的subword而不是下一最高频字节对。

实现过程：

> 1、准备足够大的训练语料
> 2、确定期望的subword词表大小
> 3、将单词拆分成字符序列
> 4、基于第3步数据训练语言模型
> 5、从所有可能的subword单元中选择加入语言模型后能最大程度地增加训练数据概率的单元作为新的单元
> 6、重复第5步直到达到第2步设定的subword词表大小或概率增量低于某一阈值

句子$S=(t_1, t_2,...,t_n)$由n个词组成，然后计算句子的似然估计值：
$$
logP(S)=\sum^{n}_{i=1}logP(t_i)
$$
如果将相邻的x，y合并，合并之后产生的词为z，之后计算句子S的似然值的变化可以表示为：
$$
logP(t_z)-(logP(t_x)+logP(t_y))=log(\frac{P(t_z)}{P(t_x)P(t_y)})
$$
因此，合并前后可以看成是两个词之间的互信息。

BERT, DistilBERT, Electra模型使用。

#### 4、ULM(Unigram Language Model)

BPE和wordPiece都是先构建小的词汇表，然后再通过合并逐步的扩充，而UML是先尽可能的构建大的词表，然后通过，通过逐步的计算，舍弃不符合要求的句子，，因此ULM考虑了句子中不同分词的可能。

> 1、初始时，建立一个足够大的词表。一般，可用语料中的所有字符加上常见的子字符串初始化词表，也可以通过BPE算法初始化。
> 2、针对当前词表，用EM算法求解每个子词在语料上的概率。
> 3、对于每个子词，计算当该子词被从词表中移除时，总的loss降低了多少，记为该子词的loss。
> 4、将子词按照Loss大小进行排序，丢弃一定比例loss最小的子词(比如20%)，保留下来的子词生成新的词表，这里需要注意的是，单字符不能被丢弃，这是为了避免OOV情况。
> 5、重复步骤2到4，直到词表大小减少到设定范围。

对于句子$$S$$，$$X=(x_1,x_2,...,x_m)$$为句子$$S$$的划分子词中的一个结果，当前分词下句子$$S$$的似然值为：
$$
P(X)=\prod \limits^m_{i=1}P(x_i)
$$
而对于句子$$S$$，需要找到似然值最大的那个划分子词的结果
$$
X^*=arg max_{x\in U(x)}P(X)
$$
其中$$U(X)$$包含了句子的所有分词结果，在实际的应用当中，词表大小由上万个，因此，不可能计算所有的分词组合，因此，这个部分可以通过维特比算法来求解得到$$X^*$$。对于$$P(x_i)$$的概率，通过EM算法来估计。

因此，ULM会保留在句子中高频率出现的子词。
$$
L=\sum^{|D|}_{s=1}log(P(X^s))=\sum^{|D|}_{s=1}log(\sum_{x\in U(X^s)}P(x))
$$
其中，$|D|$是语料库中句子的数量。因此，从上面的公式中我们可以知道，$$L$$为整个语料库的$$Loss$$。

#### 5、sentencePiece

`SentencePiece`是把一句子看做一个整体，再拆成片段，而没有保留天然的词语的概念。一般地，它把space也当做一种特殊的字符来处理，再用`BPE`或者`Unigram`算法来构造词汇表。比如，XLNetTokenizer就采用了 来代替空格，解码的时候会再用空格替换回来。目前，Tokenizers库中所有使用了`SentencePiece`的都是与`Unigram`算法联合使用的，比如ALBERT、XLNet、Marian和T5。

对于怎么去使用`sentencePiece`，可以看一下[`sentencePiece`使用介绍](https://colab.research.google.com/github/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb)。

##### 例子

~~~python
import sentencepiece as spm

# train sentencepiece model from `botchan.txt` and makes `m.model` and `m.vocab`
# `m.vocab` is just a reference. not used in the segmentation.
spm.SentencePieceTrainer.train('--input=botchan.txt --model_prefix=m --vocab_size=2000')

# makes segmenter instance and loads the model file (m.model)
sp = spm.SentencePieceProcessor()
sp.load('m.model')

# encode: text => id
print(sp.encode_as_pieces('This is a test'))
#['▁This', '▁is', '▁a', '▁t', 'est']
print(sp.encode_as_ids('This is a test'))
#[209, 31, 9, 375, 586]

# decode: id => text
print(sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']))
#This is a test
print(sp.decode_ids([209, 31, 9, 375, 586]))
#This is a test
~~~

其中'_'表示每个字符的开头。



#### 参考文章

[1] [NLP中的tokenizer介绍](https://blog.csdn.net/weixin_37447415/article/details/126583754)

[2] [NLP三大Subword模型详解](https://zhuanlan.zhihu.com/p/191648421)

[3] [NLP SubWord：理解BPE、WordPiece、ULM和BERT分词](https://zhuanlan.zhihu.com/p/401005982)

[4] [sentencePiece介绍](https://towardsdatascience.com/sentencepiece-tokenizer-demystified-d0a3aac19b15)
