# Tokenizer

## what is token？

In natural language processing tasks, ```token``` refers to the smallest unit that a machine can understand and process. For the natural language in life, in order for the machine to understand and learn the semantics, it is necessary to convert the sentence into a form that the machine can understand, and the smallest unit that can be processed by the machine is ```token```.

The process of converting the text language in life into ```token``` is ```tokenization```. This process is also called word segmentation process. According to different tokenizers, sentences can be converted into different tokens.

For ```我爱中国```, it can be divided into ```我```, ```爱```, ```中```, ```国``` and ```我```, ```爱```, ```中国```.

It can be known from this that, according to different tokenizers, the number of ```token``` after tokenization is also different. And how to divide it is better for training is what will be introduced in this article.

## Tokenizer division granularity

The tokenizer generally has three granularities, namely char, word, and subword.

**char**: A single character, such as ```a,b,c,d```.

**word**: n-gram, multiple words, etc., such as ```stuttering```.

**subword**: The subword unit, the representative form is BPE, Byte-Pair Encoding byte pair encoding, that is, a word can be split into multiple parts, for example, the word ```This``` is split into, ```Th``` 和 ```is```, for example, the tool subword-nmt.

## Advantages and disadvantages

**char**: Characters are the most basic building blocks of a language, such as ```a, b, c``` in English or ```你, 我, 他``` in Chinese. There are the following problems in using characters:

1. The number of characters is limited and usually the number is small, so when learning the embedding vector of each character, each character contains a lot of semantics, which is difficult to learn (semantic loss).
2. Divided by characters, the length of the sequence will be too long, which will greatly limit the subsequent application.

**word**: Words are the most natural language unit. For English, there are spaces in them naturally. Segmentation is relatively easy. Commonly used tokenizers include spaCy and Moses. Chinese does not have such a separator, so it is relatively difficult. However, there are also tokenizers such as Jieba, HanlP, and LTP. These tokenizers are based on rules and models, and can achieve good word segmentation results. There are two problems when using words:

1. The vocabulary is usually obtained based on the word segmentation of the corpus, but OOV(Out Of Vocabulary) may occur when encountering a new corpus.
2. The vocabulary is too large, for For the model, most of the parameters are concentrated in the input and output layers, which is not conducive to model learning, and it is easy to burst the memory (video memory). Usually the word list size does not exceed 50,000.

**subword**: It is between char and word, which can balance vocabulary size and semantic independence very well. Its segmentation criterion is that common words are not segmented, and uncommon words are segmented into subwords.

## BPE(Byte Pair Encoding)

The main purpose of byte pair encoding is for data compression. Through continuous loop iterations, pairs are paired, and the paired subwords with the highest frequency are added to the vocabulary each time until the size of the table is reached or the highest word frequency is 1.

The steps for BPE to obtain subword are as follows:

1. Prepare a sufficiently large training corpus and determine the desired Subword vocabulary size.

2. Split words into smallest units. For example, there are 26 letters in English plus various symbols, these are used as the initial vocabulary.

3. Count the frequency of adjacent unit pairs in the word on the corpus, and select the unit pair with the highest frequency to merge into a new Subword unit.

4. Repeat step 3 until the subword vocabulary set in step 1 is reached or the next highest frequency is 1.

After each merge, there may be 3 changes in the size of the vocabulary:

- 【+1】, Indicates that the merged new subwords are added, while the original 2 subwords are retained (the 2 words appear separately in the corpus).
- 【+0】, Indicates that the merged new subword is added, and at the same time, one of the original two subwords is retained and the other is eliminated (one subword appears immediately following the appearance of the other subword).
- 【-1】, Indicates that the merged new subword is added, and the original two subwords are resolved (two words appear continuously at the same time).

In fact, as the number of merges increases, the vocabulary size usually first increases and then decreases. After obtaining the subword vocabulary, it is necessary to encode the words in the sentences input into the model. The encoding process is as follows:

- Sort all subwords in the dictionary in descending order of length;
- For word ```w```, traverse the sorted dictionary in turn. Check whether the current subword is a substring of the word, and if so, output the current subword and continue to match the remaining word strings.
- If there are still substrings that do not match after traversing the dictionary, replace the remaining strings with special symbols and output, like ```nuk```.
- The representation of the word is all the above output subwords.

Decoding: If there is no stop ```</w>``` between adjacent subwords, the two subwords are directly spliced, otherwise a separator is added between the two subwords.

### Example 1

Take a sentence from a corpus as an example: ***"FloydHub is the fastest way to build, train and deploy deep learning models. Build deep learning models in the cloud. Train deep learning models."***

- Split, add suffix, count word frequency

|         WORD         | FREQUENCY |       WORD       | FREQUENCY |
| ------------------ | ------- | -------------- | ------- |
|     d e e p      |     3     |  b u i l d   |     1     |
| l e a r n i n g  |     3     |  t r a i n   |     1     |
|      t h e       |     2     |    a n d     |     1     |
|   m o d e l s    |     2     | d e p l o y  |     1     |
| F l o y d h u b  |     1     |  B u i l d   |     1     |
|       i s        |     1     | m o d e l s  |     1     |
|  f a s t e s t   |     1     |     i n      |     1     |
|      w a y       |     1     |  c l o u d   |     1     |
|       t o        |     1     |  T r a i n   |     1     |

- Create a vocabulary and count character frequencies

| NUMBER | TOKEN | FREQUENCY | NUMBER | TOKEN | FREQUENCY |
| ---- | --- | ------- | ---- | --- | ------- |
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

- After the first iteration, count the word frequency of the combination of pairwise subwords. It can be seen that the combination of ```'e'``` and ```'d'``` appears the most times, and there are 7 times in total, as shown below

|         WORD         | FREQUENCY |        WORD         | FREQUENCY |
| :---: | :---: | :---: | :---: |
|   **de** e p     |     3     |   b u i l d     |     1     |
| l e a r n i n g  |     3     |   t r a i n     |     1     |
|      t h e       |     2     |     a n d       |     1     |
| m o **de** l s   |     2     | **de** p l o y  |     1     |
| F l o y d h u b  |     1     |   B u i l d     |     1     |
|       i s        |     1     | m o **de** l s  |     1     |
|  f a s t e s t   |     1     |      i n        |     1     |
|      w a y       |     1     |   c l o u d     |     1     |
|       t o        |     1     |   T r a i n     |     1     |

- The bold place above indicates the part where ```'de'``` appears, which is 3+2+1+1=7.

Therefore, the updated vocabulary is:

|  TOKEN  |  FREQUENCY   |  TOKEN   | FREQUENCY |
| :---: | :---: | :---: | :---: |
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

The changed parts are marked in italic bold.

Continue to iterate in the follow-up, and stop iterating until the required standard is reached (the word list with the preset size or the word with the highest word frequency after merging is also 1).

Example 2

Suppose there are four words: ```low, lower, newest, widest```. They appear in the text 4, 6, 3, 5 times respectively.

~~~python
corpus:{'l o w </w>':4, 'l o w e r </w>':6, 'n e w e s t </w>':3, 'w i d e s t </w>':5}
~~~

The initial vocabulary, the vocabulary length is 11.

~~~python
Vocab:{'l','o','w','e','r','n','s','t','i','d','</w>'}
~~~

First calculate the word frequency of the combination of two or two characters. You can know that ```'lo'``` appears the most times, which is 10 times. Therefore, merge ```'l'``` and ```'o'```:

~~~python
corpus:{'lo w </w>':4, 'lo w e r </w>':6, 'n e w e s t </w>':3, 'w i d e s t </w>':5}
~~~

After merging 'l' and 'o', since ```'l'``` and ```'o'``` do not exist in the merged corpus, delete ```'l'``` and ```'o'``` in this table, and add ```'lo'```, the length of the vocabulary is 10, and the length of the vocabulary becomes shorter, corresponding to the change of ```-1``` above.

~~~python
Vocab:{'lo','w','e','r','n','s','t','i','d','</w>'}
~~~

Then continue to merge in pairs, it can be seen that ```'low'``` appears the most times, which is 10 times, so merge:

~~~python
corpus:{'low </w>':4, 'low e r </w>':6, 'n e w e s t </w>':3, 'w i d e s t </w>':5}
~~~

After the merger, because ```'lo'``` does not exist as expected, it is deleted, and ```'w'``` still exists, so it is retained. The length of the vocabulary is 10, and the length of the vocabulary remains unchanged, corresponding to ```+0 above.```change

~~~python
Vocab:{'low','w','e','r','n','s','t','i','d','</w>'}
~~~

Then continue to calculate, we can see that ```'es'``` has the highest frequency of occurrence, which is 8 times, merged:

~~~python
corpus:{'low </w>':4, 'low e r </w>':6, 'n e w es t </w>':3, 'w i d es t </w>':5}
~~~

```'e'``` continues to exist after the merger, and ```'s'``` does not exist in the corpus, so delete ```'s'``` and add ```'es'``` at the same time, the length of the vocabulary is 10, and the length of the vocabulary is 10. The length remains unchanged, corresponding to the change of ```+0``` above.

~~~python
Vocab:{'low','w','e','r','n','s','t','i','d','es','</w>'}
~~~

Call it repeatedly until the preset subword vocabulary size is reached or the frequency of the next most frequent byte pair is 1.

**Code**

~~~python
#The BPE implementation code can also be used directly using the subword-nmt package
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

To preprocess the above example:

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

result

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

### Encoding and decoding

The above process is called encoding. The decoding process is relatively simple. If there is no stop character between adjacent subwords, the two subwords are directly spliced, otherwise a separator is added between the two subwords. For example:

~~~python
# coding sequence
["the</w>", "high", "est</w>", "moun", "tain</w>"]

# decoding sequence
"the</w> highest</w> mountain</w>"
~~~

## WordPiece

The WordPiece algorithm can be seen as a variant of BPE. The difference is that WordPiece generates new subwords based on **probability** instead of the next most frequent byte pair.

Implementation process:

1. Prepare a large enough training corpus
2. Determine the desired subword vocabulary size
3. Split words into sequences of characters
4. Train language model based on step 3 data
5. Select the unit that can maximize the probability of training data after adding the language model from all possible subword units as a new unit
6. Repeat step 5 until the size of the subword vocabulary set in step 2 is reached or the probability increment is lower than a certain threshold

The sentence $S=(t_1, t_2,...,t_n)$ consists of $n$ words, and then calculate the likelihood estimate of the sentence:
$$
logP(S)=\sum^{n}_{i=1}logP(t_i)
$$
If the adjacent $x$ and $y$ are merged, the word generated after the merger is $z$, and the change of the likelihood value of the sentence $S$ after calculation can be expressed as:
$$
logP(t_z)-(logP(t_x)+logP(t_y))=log(\frac{P(t_z)}{P(t_x)P(t_y)})
$$
Therefore, before and after merging can be regarded as mutual information between two words.

Used by BERT, DistilBERT, Electra models.

## ULM(Unigram Language Model)

Both BPE and wordPiece first build a small vocabulary, and then gradually expand it through merging, while UML first builds a large vocabulary as much as possible, and then discards sentences that do not meet the requirements through step-by-step calculations, so ULM takes into account the possibility of different participle in the sentence.

1. Initially, build a sufficiently large vocabulary. Generally, all the characters in the corpus plus common substrings can be used to initialize the vocabulary, and can also be initialized by the BPE algorithm.
2. For the current vocabulary, use the EM algorithm to solve the probability of each subword on the corpus.
3. For each subword, calculate how much the total loss decreases when the subword is removed from the vocabulary, and record it as the subword's loss.
4. Sort the subwords according to the Loss size, discard a certain percentage of the subwords with the smallest loss (such as 20%), and generate a new vocabulary for the retained subwords. It should be noted here that single characters cannot be discarded. This is to avoid OOV situation.
5. Repeat steps 2 to 4 until the vocabulary size is reduced to the set range.

For the sentence $$S$$, $$X=(x_1,x_2,...,x_m)$$ is a result of the divided subwords of the sentence $$S$$, and the sentence $$S$$ under the current participle The likelihood value for is:
$$
P(X)=\prod \limits^m_{i=1}P(x_i)
$$
For the sentence $$S$$, it is necessary to find the result of dividing the subwords with the largest likelihood value
$$
X^*=arg max_{x\in U(x)}P(X)
$$
Among them, $$U(X)$$ contains all the word segmentation results of the sentence. In practical applications, the size of the vocabulary is tens of thousands. Therefore, it is impossible to calculate all the word segmentation combinations. Therefore, this part can be passed through the Viterbi algorithm. To solve to get $$X^*$$. For the probability of $$P(x_i)$$, it is estimated by the EM algorithm.

Therefore, ULM retains subwords that occur frequently in sentences.
$$
L=\sum^{|D|}_{s=1}log(P(X^s))=\sum^{|D|}_{s=1}log(\sum_{x\in U(X^s)}P(x))
$$
where $|D|$ is the number of sentences in the corpus. Therefore, we can know from the above formula that $$L$$ is the $$Loss$$ of the entire corpus.

## SentencePiece

```SentencePiece``` is to treat a sentence as a whole and then split it into fragments without retaining the concept of natural words. Generally, it treats space as a special character, and then uses the ```BPE``` or ```Unigram``` algorithm to construct the vocabulary. For example, XLNetTokenizer uses to replace spaces, and then replace them with spaces when decoding. Currently, all the Tokenizers that use ```SentencePiece``` are used in conjunction with the ```Unigram``` algorithm, such as ALBERT, XLNet, Marian, and T5.

For how to use ```sentencePiece```, you can take a look at [```sentencePiece``` introduction](https://colab.research.google.com/github/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb).

### Example

~~~python
import sentencepiece as spm

# train sentencepiece model from ```botchan.txt``` and makes ```m.model``` and ```m.vocab```
# ```m.vocab``` is just a reference. not used in the segmentation.
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

where ```'_'``` indicates the beginning of each character.

## references

[1] [NLP中的tokenizer介绍](https://blog.csdn.net/weixin_37447415/article/details/126583754)

[2] [NLP三大Subword模型详解](https://zhuanlan.zhihu.com/p/191648421)

[3] [NLP SubWord：理解BPE、WordPiece、ULM和BERT分词](https://zhuanlan.zhihu.com/p/401005982)

[4] [sentencePiece介绍](https://towardsdatascience.com/sentencepiece-tokenizer-demystified-d0a3aac19b15)
