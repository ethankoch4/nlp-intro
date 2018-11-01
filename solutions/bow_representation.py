import numpy as np


def get_bow_from_tokens(tokens, vocab):
    '''get BOW representation for text using vocab
    '''
    bow = {token: 0 for token in vocab}
    for token in tokens:
        bow[token] += 1
    return bow
    
def build_vocab(texts):
    '''build vocab dict from list of tokenized texts
    '''
    vocab = set([])
    for text in texts:
        for token in text:
            vocab.add(token)
    return vocab
    
def tokenize_text(text):
    '''tokenized an input text (str)
    '''
    tokenized = text.split()
    return tokenized
    
def build_bow_matrix(bows, vocab):
    '''build Bag-of-Words matrix using BOW dicts for each text
    '''
    
    
def texts_to_tf_matrix(texts):
    '''from list of texts to a numpy array Term-Frequency matrix
    '''
    vocab = build_vocab(texts)
    bows = []
    
    for text in texts:
        tokenized_text = tokenize_text(text)
        bow = get_bow_from_tokens(tokenized_text)
        bows.append(bow)
    
    tf_matrix = np.zeros((len(bows), len(vocab)))
    for i, bow in enumerate(bows):
        for j, token in enumerate(vocab):
            if token in bow:
                tf_matrix[i][j] = bow[token]
    return tf_matrix

def tf_matrix_to_tf_idf(tf_matrix):
    '''get a tf-idf matrix from TF matrix
    '''
    column_sums = np.sum(tf_matrix, axis=0)
    tf_idf_matrix = tf_matrix / column_sums
    return tf_idf_matrix

def get_sentiment_predictor(tf_idf, labels):
    '''return model that can be used to predict sentiment
    '''
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model = model.fit(X=tf_idf, y=labels)
    return model
    
if __name__ == '__main__:
    with open('../data/yelp_labels.csv', 'r') as f:
        labels = f.read().split(',,,,')
    with open('../data/yelp_sentences.csv', 'r') as f:
        texts = f.read().split(',,,,')
    
    
    
    
