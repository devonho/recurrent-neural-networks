import warnings
from logging import *
from functions import *

RANDOM_STATE = 50
EPOCHS = 150
BATCH_SIZE = 2048
TRAINING_LENGTH = 50
TRAIN_FRACTION = 0.7
LSTM_CELLS = 64
VERBOSE = 2
SAVE_MODEL = True

### Start #########################################################################################

warnings.filterwarnings('ignore', category=RuntimeWarning)

logging.basicConfig(filename='deepdive.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger().addHandler(logging.StreamHandler())

logging.info("**** START ****")

filters = '!"#$%&()*+/:<=>@[\\]^_`{|}~\t\n'

formatted = load_abstracts()
word_idx, idx_word, num_words, word_counts, abstracts, sequences, features, labels = make_sequences(formatted, TRAINING_LENGTH, lower=True, filters=filters)
X_train, X_valid, y_train, y_valid = create_train_valid(features, labels, num_words)

### Build model ###################################################################################
model_name = 'pre-trained-rnn'
model_dir = '../models/'

logging.info("Loading GloVe embeddings.")
embedding_matrix = load_glove_embedding_matrix(num_words)
model = make_word_level_model(
    num_words,
    embedding_matrix=embedding_matrix,
    lstm_cells=LSTM_CELLS,
    trainable=False,
    lstm_layers=1)
model.summary()

logging.info("Training model.")

callbacks = make_callbacks(model_name)
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=VERBOSE,
    callbacks=callbacks,
    validation_data=(X_valid, y_valid))

### Load model ###################################################################################

model = load_and_evaluate(model_name, X_valid, y_valid, True)
'''
# Compute frequency of each word in vocab
total_words = sum(word_counts.values())
frequencies = [word_counts[word] / total_words for word in word_idx.keys()]
frequencies.insert(0, 0)
'''

seed_html, gen_html, a_html = generate_output(model, sequences, idx_word, TRAINING_LENGTH)

logging.info(seed_html)
logging.info(gen_html)
logging.info(a_html)

logging.info("**** DONE ****")