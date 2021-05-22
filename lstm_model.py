from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
from tensorflow import keras

class LSTMModel():
    def __init__(self, mode='words'):
        '''
        mode can be either words_seq\words_seq_midi\words_seq_midi_seq
        :param mode:    if words_seq = just train using words sequences,
                        if words_seq_midi = train using words sequences + flat midi features
                        if words_seq_midi_seq = train using words sequences + flat midi features + misi sequence input
        '''
        self.mode = mode

    def build_model(self, num_words, embed_dim, embed_weights):
        # create model embedding layer

        embedding_layer = Embedding(
            num_words,
            embed_dim,
            embeddings_initializer=keras.initializers.Constant(embed_weights),
            trainable=False,
        )

        # create rest of model layers
        input = keras.Input(shape=(None,), dtype="int64")
        embedded_seq = embedding_layer(input)
        x = layers.LSTM(150)(embedded_seq)
        if self.mode == 'words_seq_midi':
            midi_feat = keras.Input(shape=(1,))
            x = Concatenate()(x, midi_feat)
            input = [input, midi_feat]
        x = layers.Dropout(0.1)(x)
        pred = layers.Dense(num_words, activation='softmax')(x)
        model = keras.Model(input, pred)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

        self.model = model
        print(model.summary())

    def train_model(self, word_seq, labels, midi_feat=None, epochs=10):
        input = word_seq
        if midi_feat is not None:
            input = [word_seq,midi_feat]
        self.model.fit(input, labels, epochs=epochs, verbose=1)