import numpy as np
from gensim.models import word2vec, fasttext

def get_vocab(corpus):
    """
    Construct the vocabulary
    :param corpus:
    :return:
    """
    pass


def apply_glove(joint_vocab, glove_path, embed_dim=200):

    wvecs_for_glove = []
    word2idx_for_glove = []
    idx2word_for_glove = []

    # set index for paddings
    pad_vec = np.zeros(embed_dim, dtype='float32')
    wvecs_for_glove.append(pad_vec)
    word2idx_for_glove.append(('<pad>', 0))
    idx2word_for_glove.append((0, '<pad>'))

    with open(glove_path, 'r','utf-8') as f:
      idx = 1
      for line in f.readlines():
        line = line.strip().split()
        if len(line) > 3:
          word, vec = line[0], list(map(float, line[1:]))
          if word in joint_vocab:
            wvecs_for_glove.append(vec)
            word2idx_for_glove.append((word, idx))
            idx2word_for_glove.append((idx, word))
            idx += 1

    wvecs_for_glove = np.array(wvecs_for_glove)
    word2idx_for_glove = dict(word2idx_for_glove)
    idx2word_for_glove = dict(idx2word_for_glove)
    return  wvecs_for_glove, word2idx_for_glove, idx2word_for_glove


def apply_skipgram(data, embed_dim=200):
    """
    Apply SkipGram embedding
    :param data: a list of sentences
    :return:
    """
    model = word2vec.Word2Vec(data, vector_size=embed_dim, min_count=1, sg=1)

    wvec_for_skipgram = []
    word2idx_for_skipgram = []
    idx2word_for_skipgram = []

    wvec_for_skipgram.append(np.zeros(embed_dim, dtype='float32'))
    word2idx_for_skipgram.append(('<pad>', 0))
    idx2word_for_skipgram.append((0, '<pad>'))

    idx = 1
    for word in model.wv.vocab.keys():
        vec = model.wv[word]
        wvec_for_skipgram.append(vec)
        word2idx_for_skipgram.append((word, idx))
        idx2word_for_skipgram.append((idx, word))
        idx += 1

    wvec_for_skipgram = np.array(wvec_for_skipgram)
    word2idx_for_skipgram = dict(word2idx_for_skipgram)
    idx2word_for_skipgram = dict(idx2word_for_skipgram)

    return wvec_for_skipgram, word2idx_for_skipgram, idx2word_for_skipgram


def apply_fasttext(data, embed_dim=200):
    model = fasttext.FastText(data, min_count=1, vector_size=embed_dim)

    wvec_for_fasttext = []
    word2idx_for_fasttext = []
    idx2word_for_fasttext = []

    wvec_for_fasttext.append(np.zeros(embed_dim, dtype='float32'))
    word2idx_for_fasttext.append(('<pad>', 0))
    idx2word_for_fasttext.append((0, '<pad>'))

    idx = 1
    for word in model.wv.vocab.keys():
        vec = model.wv[word]
        wvec_for_fasttext.append(vec)
        word2idx_for_fasttext.append((word, idx))
        idx2word_for_fasttext.append((idx, word))
        idx += 1

    wvec_for_fasttext = np.array(wvec_for_fasttext)
    word2idx_for_fasttext = dict(word2idx_for_fasttext)
    idx2word_for_fasttext = dict(idx2word_for_fasttext)

    return wvec_for_fasttext, word2idx_for_fasttext, idx2word_for_fasttext
