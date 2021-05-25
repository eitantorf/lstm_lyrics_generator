from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras

from tensorflow.keras.callbacks import EarlyStopping


class LSTMModel():
    def __init__(self, mode='words'):
        '''
        mode can be either words_seq\words_seq_midi\words_seq_midi_seq
        :param mode:    if words_seq = just train using words sequences,
                        if words_seq_midi = train using words sequences + flat midi features
                        if words_seq_midi_seq = train using words sequences + flat midi features + misi sequence input
        '''
        self.mode = mode

    def build_model(self, num_words, embed_dim, embed_weights, dropout=0.5, learning_rate=0.0001):
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
        x = layers.Dropout(dropout)(x)
        if 'midi' in self.mode:
            # add basic midi features
            midi_feat = keras.Input(shape=(259,))
            midi_feat2 = layers.Dense(20, activation='relu')(midi_feat)
            x = Concatenate()([x, midi_feat2])
            input = [input, midi_feat]
        if self.mode == 'words_seq_midi_seq':
            # add sequence midi data LSTM layer
            midi_seq_input = keras.Input(shape=(None,), dtype="float64")
            midi_seq_output = layers.LSTM(150)(midi_seq_input)
            midi_seq_output = layers.Dropout(dropout)(midi_seq_output)
            x = Concatenate()([x, midi_seq_output])
            input = [input, midi_feat, midi_seq_input]
        pred = layers.Dense(num_words, activation='softmax')(x)
        model = keras.Model(input, pred)
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)

        self.model = model
        print(model.summary())

    def train_model(self, word_seq, labels, tensorboard_callback, midi_seq=None, full_midi_sequences=None, epochs=10, validation_split=0):
        input = word_seq
        if full_midi_sequences is not None:
            input = [word_seq, midi_seq, full_midi_sequences]
        if midi_seq is not None:
            input = [word_seq,midi_seq]
        early_stopping = EarlyStopping(
                                    monitor="val_loss",
                                    min_delta=0.05,
                                    patience=8,
                                    verbose=0,
                                    mode="auto",
                                    baseline=None,
                                    restore_best_weights=False,
                                )
        self.model.fit(input, labels, callbacks=[tensorboard_callback, early_stopping], validation_split=validation_split, epochs=epochs, verbose=1)

    def predict(self, input):
        return self.model.predict(input, verbose=0).reshape(-1)