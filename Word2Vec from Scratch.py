import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils

#Function to tokenize words in a corpus, we use the Tokenizer package from Keras
def tokenize(corpus):
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(corpus)
    corpus_tokenized= tokenizer.texts_to_sequences(corpus)
    V=len(tokenizer.word_index)
    return corpus_tokenized, V   


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
    
def corpus2io(corpus_tokenized, V, window_size):
    """Converts corpus text into context and center words
    # Returns
        context and center words (arrays)
    """
    for words in corpus_tokenized:
        L = len(words)
        for index, word in enumerate(words):
            contexts = []
            labels = []
            s = index - window_size
            e = index + window_size + 1
            contexts.append([words[i]-1 for i in range(s, e) if 0 <= i < L and i != index])
            labels.append(word-1)
            x = np_utils.to_categorical(contexts, V)
            y = np_utils.to_categorical(labels, V)
            yield (x, y.ravel())
            
            
window_size = 2
corpus=["It gets easier everyday it gets easier but you gotta do it everyday that is the hard part"]
corpus_tokenized, V = tokenize(corpus)
for i, (x, y) in enumerate(corpus2io(corpus_tokenized, V, window_size)):
    print(i, "\n center word =", y, "\n context words =\n",x)
    
    
def cbow(context, label, W1, W2, loss):
    """
    Implementation of Continuous-Bag-of-Words Word2Vec model
    context: all the context words (these represent the inputs)
    label: the center word (this represents the label)
    W1: weights from the input to the hidden layer
    W2: weights from the hidden to the output layer
    loss: float that represents the current value of the loss function
    return: updated weights and loss
    """
    x = np.mean(context, axis=0)
    h = np.dot(W1.T, x)
    u = np.dot(W2.T, h)
    y_pred = softmax(u)

    e = -label + y_pred
    dW2 = np.outer(h, e)
    dW1 = np.outer(x, np.dot(W2, e))

    new_W1 = W1 - eta * dW1
    new_W2 = W2 - eta * dW2

    loss += -float(u[label == 1]) + np.log(np.sum(np.exp(u)))

    return new_W1, new_W2, loss



corpus = ["It gets easier but the hard part is you've got to do it everyday"] #our example text corpus
N = 2 #assume that the hidden layer has dimensionality = 2
window_size = 2 #symmetrical
eta = 0.1 #learning rate

corpus_tokenized, V = tokenize(corpus)

#initialize weights (with random values) and loss function
np.random.seed(100)
W1 = np.random.rand(V, N)
W2 = np.random.rand(N, V)
loss = 0.

for i, (context, label) in enumerate(corpus2io(corpus_tokenized, V, window_size)):
    W1, W2, loss = cbow(context, label, W1, W2, loss)
    print("Training example #{} \n-------------------- \n\n \t label = {}, \n \t context = {}".format(i, label, context))
    print("\t W1 = {}\n\t W2 = {} \n\t loss = {}\n".format(W1, W2, loss))