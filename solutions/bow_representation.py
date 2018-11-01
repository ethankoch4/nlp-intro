
def get_bow_from_text(tokens, vocab):
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
    
