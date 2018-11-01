# nlu-intro

This is a workshop with the goal of getting you ramped-up of a few core NLP concepts. To go through the workshop, you can go [to this link](https://colab.research.google.com/github/ethankoch4/nlu-intro/blob/master/NLP_Intro.ipynb):

https://colab.research.google.com/github/ethankoch4/nlu-intro/blob/master/NLP_Intro.ipynb


# Featured Concepts

## Tokenization

Tokenization is the act of splitting up text into logical words. In English and most Romance languages this is *almost* equivalent to splitting on whitespace.

## Bag-Of-Words

Bag-of-words (BOW) is a representation of text. It is just a map from the tokens to the number of times that token appeared in the text. Sometimes a `0` is added in for the words found in the vocab, but not in the text.

## Term-Frequency Matrix

This simply counts the frequency of the terms from the vocab found in the text. The rows of this matrix correspond to each document, the columns are tokens, and the entries are the number of times that token appeared in the document. This is simply to compute from BOW.

## Term-Frequency Inverse-Document-Frequency

Highly frequent terms are often stop-words that don't help with the task at hand. So, to essentially down-weight the highly frequent words, we divide each entry in the TF matrix with the number of non-zero entries in its column.

## Sentiment Classification

It's often of interest to identify positivity/negativity in text. This is called Sentiment Classification or Sentiment Analysis. Sometimes people use a `[-1, 1]` scale, sometimes they use `[0, 1]`, with neutral being implicitly somewhere in the middle.
