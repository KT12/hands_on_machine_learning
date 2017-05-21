# Embeddings
print('Build word embeddings')

import tensorflow as tf
import numpy as np
from six.moves import urllib
import errno
import os
import zipfile
from collections import Counter, deque
import random

random.seed(5)
tf.set_random_seed(5)

WORDS_PATH = 'datasets/words'
WORDS_URL = 'http://mattmahoney.net/dc/text8.zip'

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)

def fetch_words_data(words_url=WORDS_URL, words_path=WORDS_PATH):
    os.makedirs(words_path, exist_ok=True)
    zip_path = os.path.join(words_path, 'words.zip')
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(words_url, zip_path)
    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])
    return data.decode('ascii').split()  

words = fetch_words_data()
print(words[:5])

# Build Dictionary
print('Buliding Dictionary')

vocabulary_size = 25000

vocabulary = [("UNK", None)] + Counter(words).most_common(vocabulary_size - 1)
vocabulary = np.array([word for word, _ in vocabulary])
dictionary = {word: code for code, word in enumerate(vocabulary)}
data = np.array([dictionary.get(word, 0) for word in words])

print(' '.join(words[:9]), data[:9])
print(' '.join([vocabulary[word_index] \
    for word_index in [5234, 3081, 12, 6, 195, 2, 3134, 46, 59]]))
print(words[24], data[24])

# Generate batches
print('Generating batches')

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

data_index = 0
batch, labels = generate_batch(8, 2, 1)
print(batch, [vocabulary[word] for word in batch])
print(labels, [vocabulary[word] for word in labels[:,0]])

# Build the model
print('Building model')

batch_size = 128
embedding_size = 128 # Dim of embedding vector
skip_window = 1 # How many words to look at on left and right
num_skips = 2 # Times to reuse an input to generate label

# Use random validation set
# Validation samples limited to words with low numeric ID
# This ensures they appear frequently
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

learning_rate = 0.01

# Input data
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Look up embeddings for inputs
init_embeddings = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
embeddings = tf.Variable(init_embeddings)
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Construct the variables for the NCE loss
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
        stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Compute the avg NCE loss for the batch
# tf.nce_loss automatically draws a new sample of negative labels
# each time the loss is evaluated

loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, train_labels,
    embed, num_sampled, vocabulary_size))

# Construct AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# Compute the cosine similarity between minibatch examples and all embeddings
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
normalized_embeddings = embeddings /norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

init = tf.global_variables_initializer()

# Train the model
print('Training model')

num_steps = 10001
with tf.device("/cpu:0"):

  with tf.Session() as sess:
    init.run()

    avg_loss = 0
    for step in range(num_steps):
        print("\rIteration: {}".format(step), end='\t')
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # Perform 1 update step by evaluating the training op 
        # (including it in the list of returned vals for session.run())
        _, loss_val = sess.run([training_op, loss], feed_dict=feed_dict)
        avg_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                avg_loss /=2000
            print('Avg loss at step ', step, ': ', avg_loss)
            avg_loss = 0
        
        # Next op is expensive
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = vocabulary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = vocabulary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()