# run hypothesis test experiment for a given file, calculating the mean associations
# Credits: Stanford NLP GloVe project

# proving racial bias in each region separately
# Step 1: load the model
# Step 2: calculate mean associations for names to pos/neg, how do they differ 
# Step 3: calculate randomized hypothesis test mean associations (do this like 1000 times)
# Step 4: plot on a curve, show that there is racial bias (this is the first step)

# proving changes across regions
# Step 1: create 15 randomized corpora...15 choose 2 is 105? Enough data pts for a distribution
# Step 2: generate the embeddings
# Step 3: do the same test as above, get a test statistic to work with. Note, we need to normalize
# Step 4: calculate all combinations of differences, this yields a distribution.
# Step 5: plot the differences we care about on that actual distribution.
# Step 6: example for time period: differences 4-3, 4-2, 4-1...do we see a pattern? or is it all over?


# what are the differences to compare?
# Need to compare all regions and time periods against each other: 435 comparisons
# number of long-term charts: 5 regions over time, and then 6 time periods split over region...11 total
# how many combinatorial baselines do I need? Perhaps 25 choose 2?
# how to create those sets: just seek to a random line in each, and then take a random amount...goal is to 
# make the file sizes relatively similar


import argparse
import numpy as np
import spacy
import random
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange
from statistics import NormalDist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab.txt', type=str)
    parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    parser.add_argument('--randomized_trials', nargs=1, metavar=('NUM_TRIALS'))
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

    title = ' '.join(args.vocab_file[6:-4].split('_')[:-1]) # use this for not randomized
    # title = args.vocab_file[6:-4] # only use this for randomized
    print(title)

    # note: we can use W_norm for all normalized
    # we first remove the OOV words before randomizing, because for this dataset the OOV words are not relevant
    pleasant, unpleasant = parse_words()
    pleasant_rev, unpleasant_rev = remove_oov_words(W_norm, vocab, pleasant), remove_oov_words(W_norm, vocab, unpleasant)
    black_names, white_names = parse_names()
    # black_names, white_names = remove_oov_words(W_norm, vocab, black_names), remove_oov_words(W_norm, vocab, white_names)

    test_statistics = []
    if args.randomized_trials:
        for _ in trange(int(args.randomized_trials[0])):
            curr_pleasant, curr_unpleasant = randomize_categories(pleasant_rev, unpleasant_rev)
            curr_stat = calculate_test_statistic(
                W_norm, vocab, white_names, black_names, curr_pleasant, curr_unpleasant)
            test_statistics.append(curr_stat)

    test_statistic = calculate_test_statistic(W_norm, vocab, white_names, black_names, pleasant_rev, unpleasant_rev)
    print('Test statistic for the given data', test_statistic)

    all_test_statistics = np.array(test_statistics)
    plot_data(all_test_statistics, test_statistic, title)

def randomize_categories(pleasant_orig, unpleasant_orig):
    '''
    Given the original pleasant and unpleasant lists, scramble them (for randomization trials)  
    '''
    all_words = pleasant_orig + unpleasant_orig
    pleasant_shuffled = []
    unpleasant_shuffled = []

    for word in all_words:
        curr = random.randint(0, 1)
        if curr == 0:
            pleasant_shuffled.append(word)
        else:
            unpleasant_shuffled.append(word)

    return pleasant_shuffled, unpleasant_shuffled

def calculate_test_statistic(W, vocab, white_names, black_names, pleasant, unpleasant):
    '''
    Run the test statistic calculation for a set of white and black names, and a set of pleasant and
    unpleasant words.
    :param W: the vector embeddings matrix
    :param vocab: the vocabulary in the GloVe embeddings
    :param white_names: list of white names
    :param black_names: list of black names
    :param pleasant: list of pleasant words
    :param unpleasant: list of unpleasant words
    '''
    sum_white_names = 0
    sum_black_names = 0

    for name in white_names:
        curr_mean = calculate_mean_name_association(
            W, vocab, name, pleasant, unpleasant)
        sum_white_names += curr_mean

    for name in black_names:
        curr_mean = calculate_mean_name_association(
            W, vocab, name, pleasant, unpleasant)
        sum_black_names += curr_mean
    return sum_white_names - sum_black_names

def remove_oov_words(W, vocab, check_words):
    '''
    Helper function to determine which words are out of vocabulary, and return only them
    Prints out the number of words that are considered OOV
    :param W: the vector embeddings matrix
    :param vocab: the vocabulary in the GloVe embeddings
    :param check_words: the set of words to check against the GloVe vocabulary
    '''
    output_words = [word for word in check_words if get_embedding(W, vocab, word) is not None]
    return output_words
    

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

    name_embedding = get_embedding(W, vocab, name)
    if name_embedding is None:
        return 0

    for i in range(len(pleasant)):
        pleasant_embedding = get_embedding(W, vocab, pleasant[i])
        means_pleasant[i] = cosine_similarity(name_embedding, pleasant_embedding)

    for i in range(len(unpleasant)):
        unpleasant_embedding = get_embedding(W, vocab, unpleasant[i])
        means_unpleasant[i] = cosine_similarity(name_embedding, unpleasant_embedding)
    return np.mean(means_pleasant) - np.mean(means_unpleasant)

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
    cosine_sim = numerator / denominator

    return cosine_sim

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
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    black = []
    white = []
    with open('../f21_iw/black_butler.txt', 'r') as black_names:
        black_orig = black_names.readlines()
        for name in black_orig:
            black.append(" ".join([token.lemma_ for token in nlp(name[:-1].lower())]))
    with open('../f21_iw/white_butler.txt', 'r') as white_names:
        white_orig = white_names.readlines()
        for name in white_orig:
            white.append(
                " ".join([token.lemma_ for token in nlp(name[:-1].lower())]))
    return black, white

def plot_data(randomized_data, actual_value, title):
    '''
    Takes an array of values
    '''
    mu, std = np.mean(randomized_data), np.std(randomized_data)
    curr_dist = NormalDist(mu=mu, sigma=std)
    critical_val = curr_dist.inv_cdf(0.95)
    print('Mean: ' + str(curr_dist.mean))
    print('Std Dev: ' + str(curr_dist.stdev))

    sns.set_style('whitegrid')
    sns.kdeplot(randomized_data, bw=0.5).set_title(title)
    plt.axvline(x=actual_value)
    plt.axvline(x=critical_val, linestyle='--')
    plt.savefig('_'.join(title.split(' ')) + '_butler.png')

if __name__ == '__main__':
    main()
