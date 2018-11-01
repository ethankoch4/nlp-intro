
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
    
    
def texts_to_bow_matrix(texts):
    '''from list of texts to a numpy array Bag-of-Words matrix
    '''
    vocab = build_vocab(texts)
    bows = []
    
    for text in texts:
        tokenized_text = tokenize_text(text)
        bow = get_bow_from_tokens(tokenized_text)
        bows.append(bow)
    
    bow_matrix = np.zeros((len(bows), len(vocab)))
    for i, bow in enumerate(bows):
        for j, token in enumerate(vocab):
            if token in bow:
                bow_matrix[i][j] = bow[token]
    return bow_matrix
