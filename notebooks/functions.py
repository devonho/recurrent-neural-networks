import warnings
import logging
import pandas as pd
import numpy as np
import re
import os
import gc
import random

from tensorflow.python.client import device_lib

from keras.preprocessing.text import Tokenizer
from keras.utils import get_file
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ProgbarLogger, Callback

from sklearn.utils import shuffle

#from utils import addContent

RANDOM_STATE = 50
EPOCHS = 150
BATCH_SIZE = 2048
TRAINING_LENGTH = 50
TRAIN_FRACTION = 0.7
LSTM_CELLS = 64
VERBOSE = 0
SAVE_MODEL = True

model_name = 'pre-trained-rnn'
model_dir = '../models/'

def show_tensorflow_config():
    '''Show devices used by Tensorflow'''
    device_lib.list_local_devices()

def load_abstracts():
    '''Read in CSV, remove images/citations, return as list'''
    # Read in data
    data = pd.read_csv('../data/neural_network_patent_query.csv', parse_dates=['patent_date'])

    # Extract abstracts
    original_abstracts = list(data['patent_abstract'])

    # Iterate through all the original abstracts
    formatted = []
    for a in original_abstracts:
        formatted.append(format_patent(a))
    return formatted

def format_patent(patent):
    """Add spaces around punctuation and remove references to images/citations."""

    # Add spaces around punctuation
    patent = re.sub(r'(?<=[^\s0-9])(?=[.,;?])', r' ', patent)

    # Remove references to figures
    patent = re.sub(r'\((\d+)\)', r'', patent)

    # Remove double spaces
    patent = re.sub(r'\s\s', ' ', patent)
    return patent

def make_sequences(texts,
                   training_length=50,
                   lower=True,
                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    """Turn a set of texts into sequences of integers"""

    # Create the tokenizer object and train on texts
    tokenizer = Tokenizer(lower=lower, filters=filters)
    tokenizer.fit_on_texts(texts)

    # Create look-up dictionaries and reverse look-ups
    word_idx = tokenizer.word_index
    idx_word = tokenizer.index_word
    num_words = len(word_idx) + 1
    word_counts = tokenizer.word_counts

    logging.info(f'There are {num_words} unique words.')

    # Convert text to sequences of integers
    sequences = tokenizer.texts_to_sequences(texts)

    # Limit to sequences with more than training length tokens
    seq_lengths = [len(x) for x in sequences]
    over_idx = [i for i, l in enumerate(seq_lengths) if l > (training_length + 20)]
    new_texts = []
    new_sequences = []

    # Only keep sequences with more than training length tokens
    for i in over_idx:
        new_texts.append(texts[i])
        new_sequences.append(sequences[i])

    training_seq = []
    labels = []

    # Iterate through the sequences of tokens
    for seq in new_sequences:
        # Create multiple training examples from each sequence
        for i in range(training_length, len(seq)):
            # Extract the features and label
            extract = seq[i - training_length:i + 1]

            # Set the features and label
            training_seq.append(extract[:-1])
            labels.append(extract[-1])

    logging.info(f'There are {len(training_seq)} training sequences.')

    # Return everything needed for setting up the model
    return word_idx, idx_word, num_words, word_counts, new_texts, new_sequences, training_seq, labels

def find_answer(index):
    """Find label corresponding to features for index in training data"""

    # Find features and label
    feats = ' '.join(idx_word[i] for i in features[index])
    answer = idx_word[labels[index]]

    logging.info('Features:', feats)
    logging.info('\nLabel: ', answer)

def create_train_valid(features,
                       labels,
                       num_words,
                       train_fraction=TRAIN_FRACTION):
    """Create training and validation features and labels."""

    # Randomly shuffle features and labels
    features, labels = shuffle(features, labels, random_state=RANDOM_STATE)

    # Decide on number of samples for training
    train_end = int(train_fraction * len(labels))

    train_features = np.array(features[:train_end])
    valid_features = np.array(features[train_end:])

    train_labels = labels[:train_end]
    valid_labels = labels[train_end:]

    # Convert to arrays
    X_train, X_valid = np.array(train_features), np.array(valid_features)

    # Using int8 for memory savings
    y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
    y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)

    # One hot encoding of labels
    for example_index, word_index in enumerate(train_labels):
        y_train[example_index, word_index] = 1

    for example_index, word_index in enumerate(valid_labels):
        y_valid[example_index, word_index] = 1

    # Memory management
    import gc
    gc.enable()
    del features, labels, train_features, valid_features, train_labels, valid_labels
    gc.collect()

    return X_train, X_valid, y_train, y_valid

def find_closest(query, embedding_matrix, word_idx, idx_word, n=10):
    """Find closest words to a query word in embeddings"""

    idx = word_idx.get(query, None)
    # Handle case where query is not in vocab
    if idx is None:
        logging.info(f'{query} not found in vocab.')
        return
    else:
        vec = embedding_matrix[idx]
        # Handle case where word doesn't have an embedding
        if np.all(vec == 0):
            logging.info(f'{query} has no pre-trained embedding.')
            return
        else:
            # Calculate distance between vector and all others
            dists = np.dot(embedding_matrix, vec)

            # Sort indexes in reverse order
            idxs = np.argsort(dists)[::-1][:n]
            sorted_dists = dists[idxs]
            closest = [idx_word[i] for i in idxs]

    logging.info(f'Query: {query}\n')
    max_len = max([len(i) for i in closest])
    # Print out the word and cosine distances
    for word, dist in zip(closest, sorted_dists):
        logging.info(f'Word: {word:15} Cosine Similarity: {round(dist, 4)}')

def make_word_level_model(num_words,
                          embedding_matrix,
                          lstm_cells=64,
                          trainable=False,
                          lstm_layers=1,
                          bi_direc=False):
    """Make a word level recurrent neural network with option for pretrained embeddings
       and varying numbers of LSTM cell layers."""

    model = Sequential()

    # Map words to an embedding
    if not trainable:
        model.add(
            Embedding(
                input_dim=num_words,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=False,
                mask_zero=True))
        model.add(Masking())
    else:
        model.add(
            Embedding(
                input_dim=num_words,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=True))

    # If want to add multiple LSTM layers
    if lstm_layers > 1:
        for i in range(lstm_layers - 1):
            model.add(
                LSTM(
                    lstm_cells,
                    return_sequences=True,
                    dropout=0.1,
                    recurrent_dropout=0.1))

    # Add final LSTM cell layer
    if bi_direc:
        model.add(
            Bidirectional(
                LSTM(
                    lstm_cells,
                    return_sequences=False,
                    dropout=0.1,
                    recurrent_dropout=0.1)))
    else:
        model.add(
            LSTM(
                lstm_cells,
                return_sequences=False,
                dropout=0.1,
                recurrent_dropout=0.1))
    model.add(Dense(128, activation='relu'))
    # Dropout for regularization
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_words, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

def make_callbacks(model_name, save=SAVE_MODEL):
    """Make list of callbacks for training"""

    class CallBackLogger(Callback):
        def on_epoch_begin(self, epoch):
            logging.info('Epoch began ' + epoch)

        def on_epoch_end(self, epoch):
            logging.info('Epoch ended ' + epoch)


    callbacks = [EarlyStopping(monitor='val_loss', patience=5), ProgbarLogger(), CallBackLogger()]

    if save:
        callbacks.append(
            ModelCheckpoint(
                f'{model_dir}{model_name}.h5',
                save_best_only=True,
                save_weights_only=False))
    return callbacks

def load_and_evaluate(model_name, X_valid, y_valid, return_model, evaluate=False):
    """Load in a trained model and evaluate with log loss and accuracy"""

    model = load_model(f'{model_dir}{model_name}.h5')
    
    if evaluate:
        r = model.evaluate(X_valid, y_valid, batch_size=2048, verbose=1)

        valid_crossentropy = r[0]
        valid_accuracy = r[1]

        logging.info(model.metrics_names)
        logging.info(f'Cross Entropy: {round(valid_crossentropy, 4)}')
        logging.info(f'Accuracy: {round(100 * valid_accuracy, 2)}%')

    if return_model:
        return model

def load_glove_embedding_matrix(num_words):
    # Vectors to use
    glove_vectors = '../datasets/glove.6B.zip'

    # Download word embeddings if they are not present
    if not os.path.exists(glove_vectors):
        glove_vectors = get_file('glove.6B.zip',
                                'http://nlp.stanford.edu/data/glove.6B.zip')
        os.system(f'unzip {glove_vectors}')

    # Load in unzipped file
    glove_vectors = '../datasets/glove.6B.100d.txt'
    glove = np.loadtxt(glove_vectors, dtype='str', comments=None)

    vectors = glove[:, 1:].astype('float')
    words = glove[:, 0]
    del glove
    
    word_lookup = {word: vector for word, vector in zip(words, vectors)}

    embedding_matrix = np.zeros((num_words, vectors.shape[1]))

    '''
    not_found = 0
    for i, word in enumerate(word_idx.keys()):
        # Look up the word embedding
        vector = word_lookup.get(word, None)

        # Record in matrix
        if vector is not None:
            embedding_matrix[i + 1, :] = vector
        else:
            not_found += 1

    logging.info(f'There were {not_found} words without pre-trained embeddings.')
    '''
    gc.enable()
    del vectors
    gc.collect()

    # Normalize and convert nan to 0
    embedding_matrix = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1).reshape((-1, 1))
    embedding_matrix = np.nan_to_num(embedding_matrix)
    return embedding_matrix

def generate_output(model,
                    sequences,
                    idx_word,
                    training_length=50,
                    new_words=50,
                    diversity=1,
                    return_output=False,
                    n_gen=1):
    """Generate `new_words` words of output from a trained model and format into HTML."""

    # Choose a random sequence
    seq = random.choice(sequences)

    # Choose a random starting point
    seed_idx = random.randint(0, len(seq) - training_length - 10)
    # Ending index for seed
    end_idx = seed_idx + training_length

    gen_list = []

    for n in range(n_gen):
        # Extract the seed sequence
        seed = seq[seed_idx:end_idx]
        original_sequence = [idx_word[i] for i in seed]
        generated = seed[:] + ['#']

        # Find the actual entire sequence
        actual = generated[:] + seq[end_idx:end_idx + new_words]

        # Keep adding new words
        for i in range(new_words):

            # Make a prediction from the seed
            preds = model.predict(np.array(seed).reshape(1, -1))[0].astype(
                np.float64)

            # Diversify
            preds = np.log(preds) / diversity
            exp_preds = np.exp(preds)

            # Softmax
            preds = exp_preds / sum(exp_preds)

            # Choose the next word
            probas = np.random.multinomial(1, preds, 1)[0]

            next_idx = np.argmax(probas)

            # New seed adds on old word
            seed = seed[1:] + [next_idx]
            generated.append(next_idx)

        # Showing generated and actual abstract
        n = []

        for i in generated:
            n.append(idx_word.get(i, '< --- >'))

        gen_list.append(n)

    a = []

    for i in actual:
        a.append(idx_word.get(i, '< --- >'))

    a = a[training_length:]

    gen_list = [
        gen[training_length:training_length + len(a)] for gen in gen_list
    ]

    if return_output:
        return original_sequence, gen_list, a

    # HTML formatting
    '''
    seed_html = ''
    seed_html = addContent(seed_html, header('Seed Sequence', color='darkblue'))
    seed_html = addContent(seed_html,box(remove_spaces(' '.join(original_sequence))))

    gen_html = ''
    gen_html = addContent(gen_html, header('RNN Generated', color='darkred'))
    gen_html = addContent(gen_html, box(remove_spaces(' '.join(gen_list[0]))))

    a_html = ''
    a_html = addContent(a_html, header('Actual', color='darkgreen'))
    a_html = addContent(a_html, box(remove_spaces(' '.join(a))))
    
    return seed_html, gen_html, a_html    
    '''
    return ' '.join(original_sequence), ' '.join(gen_list[0]), ' '.join(a)

if __name__ == '__main__':
    logging.info(show_tensorflow_config())
    #logging.info(load_abstracts())