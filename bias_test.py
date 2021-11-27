# run hypothesis test experiment for a given file, calculating the mean associations
# Credits: Stanford NLP GloVe project

# proving racial bias in each region separately
# Step 1: load the model
# Step 2: calculate mean associations for names to pos/neg, how do they differ 
# Step 3: calculate randomized hypothesis test mean associations (do this like 1000 times)
# Step 4: plot on a curve, show that there is racial bias (this is the first step)

# proving changes across regions
# Step 1: create 15 randomized corpora
# Step 2: generate the embeddings
# Step 3: due the same test as above, get a test statistic to work with
# Step 4: calculate all combinations of differences
# Step 5: plot the differences we care about on that actual distribution


import argparse
import numpy as np
import spacy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab.txt', type=str)
    parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    args = parser.parse_args()

    with open(args.vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit length
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T

    # note: we can use W_norm for all normalized
    pleasant, unpleasant = parse_words()
    black_names, white_names = parse_names()
    res, error = calculate_mean_name_association(W, vocab, 'john', pleasant, unpleasant)
    print(error)

def calculate_mean_name_association(W, vocab, name, pleasant, unpleasant):
    '''
    Helper function to return word association s(w, A, B) for a word and pleasant/unpleasant sets
    :param W: the vector embeddings matrix
    :param vocab: the vocabulary in the GloVe embeddings
    :param name: name to use for comparison
    :param pleasant: set of pleasant words
    :param unpleasant: set of unpleasant words
    '''
    means_pleasant = np.zeros(len(pleasant))
    means_unpleasant = np.zeros(len(unpleasant))
    error_words = []

    split_size = 100

    name_embedding = get_embedding(W, vocab, name)
    # Name not found error. TODO: handle!
    if name_embedding is None:
        return 0

    for i in range(len(pleasant)):
        pleasant_embedding = get_embedding(W, vocab, pleasant[i])
        if pleasant_embedding is None:
            error_words.append(pleasant[i])
            continue
        means_pleasant[i] = cosine_similarity(name_embedding, pleasant_embedding)

    for i in range(len(unpleasant)):
        unpleasant_embedding = get_embedding(W, vocab, unpleasant[i])
        if unpleasant_embedding is None:
            error_words.append(unpleasant[i])
            continue
        means_unpleasant[i] = cosine_similarity(name_embedding, unpleasant_embedding)
    
    print(np.mean(means_pleasant) - np.mean(means_unpleasant))
    print(f'Number of error words: {len(error_words)}')
    print('*****')
    return np.mean(means_pleasant) - np.mean(means_unpleasant), error_words

# get the vector embedding for a given word
def get_embedding(W, vocab, word):
    index = vocab.get(word, None)
    if index is not None:
        return W[index, :]
    return None


# cite: https://www.kaggle.com/cdabakoglu/word-vectors-cosine-similarity
def cosine_similarity(a, b):
    numerator = np.dot(a, b)

    a_norm = np.sqrt(np.sum(a**2))
    b_norm = np.sqrt(np.sum(b**2))

    denominator = a_norm * b_norm
    cosine_similarity = numerator / denominator

    return cosine_similarity

def parse_words():
    # if the word has a negative sign, it's unpleasant. Otherwise, pleasant.
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    pleasant = []
    unpleasant = []
    with open('../f21_iw/AFINN-111.txt', 'r') as input_words:
        curr_line = input_words.readline()
        while curr_line:
            temp = curr_line.split()
            word = ' '.join(temp[:-1])
            val = int(temp[-1])

            formatted_word = " ".join([token.lemma_ for token in nlp(word)])

            if val > 0:
                pleasant.append(formatted_word)
            else:
                unpleasant.append(formatted_word)

            curr_line = input_words.readline()
    return list(set(pleasant)), list(set(unpleasant))

def parse_names():
    black = []
    white = []
    print('got this far!')
    with open('../f21_iw/black_names.txt', 'r') as black_names:
        black = black_names.readlines()
        black = [x[:-1].lower() for x in black] # remove newline character
    
    with open('../f21_iw/black_names.txt', 'r') as white_names:
        white = white_names.readlines()
        white = [x[:-1].lower() for x in white] # remove newline character
    
    return black, white

main()
        


