# Word2Vec-representation
**Project still messy**

Implemented  Word2Vec  representation  of  words  using  back-propagation  algorithm.The objective of Word2Vec is to find low-dimensional representations of words.  Themodel  successfully  mapped  the  input  words  to  a  low  dimensional  vector  space  suchthat similar words have similar representations.

In word2vec we scan through a text corpus and for each training example we define a center word with its surrounding context words. Depending on the algorithm of choice (Continuous Bag-of-Words or Skip-gram), the center and context words may work as inputs and labels, respectively, or vice versa.

Typically the context words are defined as a symmetric window of predefined length, on both the left and right hand sides of the center word. Also, let’s say that we define our window to be symmetric around the center word and of length two. Then, our one-hot encoded context and center words can be visualized as follows,

In the CBOW model the input is represented by the context words and the labels (ground truth) by the center words.

The code was inspired by the Stanford NLP course, where Richard Socher jots down the entire math of the backpropogation algorithm!
Inspired me to attempt to code the same.

The model was trained on a tiny corpus due to lack of computational power, but the if the same model is applied to a large corpus it should provide more desirable and deterministic results.

## Structure

Word2Vec from Scratch.py - Naive Implementation(Too much space)
ipynb - Larger vocab + Space Efficient, mini batch training pending




