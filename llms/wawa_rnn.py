import tensorflow as tf

import os

from one_step import OneStep


# RNN model from Tensorflow's guide to text generation with RNNs
class WawaRnn(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x


def text_from_ids(ids, chars_from_ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1).numpy().decode('UTF-8')


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def create_model(ids_from_chars, embedding_dim=256, rnn_units=1024):
    return WawaRnn(vocab_size=len(ids_from_chars.get_vocabulary()),
                   embedding_dim=embedding_dim,
                   rnn_units=rnn_units)


def train_rnn(ids_from_chars, text):
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    
    seq_length = 512
    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    EPOCHS = 10

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    model = create_model(ids_from_chars)
    model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))
    checkpoint_dir = './llms/rnn_training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "check_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    
    
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
    
    return history


def load_rnn(checkpoint_name, model, chars_from_ids, ids_from_chars):
    model.load_weights(checkpoint_name).expect_partial()
    return OneStep(model, chars_from_ids, ids_from_chars)


def generate_text(one_step_model, seed_text='雨が', n=512):
    states = None
    next_char = tf.constant([f'{seed_text}'])
    result = [next_char]

    for n in range(n):
      next_char, states = one_step_model.generate_one_step(next_char, states=states)
      result.append(next_char)

    result = tf.strings.join(result)
    return result[0].numpy().decode('UTF-8')
  

def rnn_generate_text(keywords):
    # Given keywords to start the prompt, load the RNN model and generate text
    with open('./lyrics/lyrics_cleaned_rnn.txt', 'r', encoding='UTF-8') as reader:
        text = '\n'.join(reader.readlines())

    vocab = sorted(set(text))
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

    model = create_model(ids_from_chars)
    checkpoint_dir = './llms/rnn_training_checkpoints'
    checkpoint = os.path.join(checkpoint_dir, "check_10")

    rnn = load_rnn(checkpoint, model, chars_from_ids, ids_from_chars)
    return generate_text(rnn, seed_text=keywords)


def main():
    print(rnn_generate_text(input('Enter Japanese seed text: ')))


if __name__ == '__main__':
    main()
