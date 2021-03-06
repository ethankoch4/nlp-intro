import numpy as np


def get_bow_from_tokens(tokens, vocab):
    '''get BOW representation for text using vocab
    
    Arguments:
        tokens (list): list of strings (tokens) from a text
        vocab (set): all tokens in corpus
    
    Returns:
        bag-of-words representation of tokens passed in
    '''
    bow = {token: 0 for token in vocab}
    for token in tokens:
        bow[token] += 1
    return bow

def build_vocab(texts):
    '''build vocab dict from list of tokenized texts
    
    Arguments:
        texts (list): list of lists of strings (tokens) from each text
    
    Returns:
        set of each unique token in corpus
    '''
    vocab = set([])
    for text in texts:
        for token in text:
            vocab.add(token)
    return vocab

def tokenize_text(text):
    '''tokenized an input text (str)
    
    Arguments:
        text (str): one of the documents in the corpus
    
    Returns:
        list of tokens (strings) in document
    '''
    tokenized = text.split()
    return tokenized

def texts_to_tf_matrix(texts):
    '''from list of texts to a numpy array Term-Frequency matrix
    
    Arguments:
        texts (list): list of strings that are the input texts
    
    Returns:
        Term-Frequency matrix -- rows are texts, columns are words
    '''
    vocab = build_vocab([tokenize_text(text) for text in texts])
    bows = []

    for text in texts:
        tokenized_text = tokenize_text(text)
        bow = get_bow_from_tokens(tokenized_text, vocab)
        bows.append(bow)

    tf_matrix = np.zeros((len(bows), len(vocab)))
    for i, bow in enumerate(bows):
        for j, token in enumerate(vocab):
            if token in bow:
                tf_matrix[i][j] = bow[token]
    return tf_matrix

def tf_matrix_to_tf_idf(tf_matrix):
    '''get a tf-idf matrix from TF matrix
    
    Arguments:
        tf_matrix (np.array): np.array that is a Term-Frequency matrix
    
    Returns:
        Term-Frequency Inverse-Document matrix
    '''
    column_sums = np.count_nonzero(tf_matrix, axis=0)
    tf_idf_matrix = tf_matrix / column_sums
    return tf_idf_matrix

def get_sentiment_predictor(matrix, labels):
    '''return model that can be used to predict sentiment
    
    Arguments:
        matrix (np.array): (X) examples to train classifier on
        labels (list): (Y) labels to train classifier on
    
    Returns:
        sklearn.linear_model.LogisticRegression object that can be used to predict
    '''
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model = model.fit(X=matrix, y=labels)
    return model

if __name__ == '__main__':
    with open('../data/yelp_labels.csv', 'r') as f:
        labels = f.read().split(',,,,')
    with open('../data/yelp_sentences.csv', 'r') as f:
        texts = f.read().split(',,,,')

    train_labels = labels[:100]
    test_labels = labels[-100:]


    tf_matrix = texts_to_tf_matrix(texts)
    train_tf = tf_matrix[:100]
    test_tf = tf_matrix[-100:]
    tf_sent_predictor = get_sentiment_predictor(train_tf, train_labels)
    tf_sent_accuracy = tf_sent_predictor.score(test_tf, test_labels)
    print('Performance using TF matrix: {0}'.format(round(tf_sent_accuracy, 4)))

    tf_idf = tf_matrix_to_tf_idf(tf_matrix)
    train_tf_idf = tf_idf[:100]
    test_tf_idf = tf_idf[-100:]
    tf_idf_sent_predictor = get_sentiment_predictor(train_tf_idf, train_labels)
    tf_idf_sent_accuracy = tf_sent_predictor.score(test_tf_idf, test_labels)
    print('Performance using TF-IDF matrix: {0}'.format(round(tf_idf_sent_accuracy, 4)))
