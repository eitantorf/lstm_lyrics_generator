import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import os
from tqdm import tqdm

IS_COLAB = (os.name == 'posix')

if IS_COLAB:
    import lstm_lyrics_generator.midi_feature_extractor as mfe
else:
    import midi_feature_extractor as mfe

class SongHandler():
    def __init__(self, lyrics_list, max_sequence_len = 20):
        # init tokenizer based on input lyrics
        max_sequence_len = max_sequence_len
        tokenizer = Tokenizer(filters='!"#$%()*+,-./:;<=>?@[\\]^_`{|}~\t\n')  # removed &, so it's kept as a valid token
        tokenizer.fit_on_texts(lyrics_list)
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len

    def generate_train_sequences(self, songs_lyrics, midi_features=None):
        tokened_lyrics = self.tokenizer.texts_to_sequences(songs_lyrics)
        train_sequences = []
        midi_sequence = []
        for k, lyric in enumerate(tqdm(tokened_lyrics)):
            for i in range(1, len(lyric)):
                passed_max = 1 if i >= self.max_sequence_len else 0
                seq = lyric[passed_max*(i-self.max_sequence_len+1):i+1]
                train_sequences.append(seq)
                if midi_features is not None:
                    midi_sequence.append(midi_features[k, :])
        #pad sequences
        train_sequences_pad = np.array(pad_sequences(train_sequences,maxlen=self.max_sequence_len, padding='pre'))
        sequences, label = train_sequences_pad[:,:-1],train_sequences_pad[:,-1]
        if midi_features is not None:
            return sequences, midi_sequence, label
        else:
            return sequences, label

    def generate_train_sequences_with_midi(self, songs_lyrics, songs_durations, midi_features, pm_list):
        tokened_lyrics = self.tokenizer.texts_to_sequences(songs_lyrics)
        train_sequences = []
        midi_sequence = []
        full_midi_sequence = []
        for k, lyric in enumerate(tqdm(tokened_lyrics)):
            num_words = len(lyric)
            sec_per_word = (songs_durations[k] - 20) / num_words  # remove 10 second intro and 10 second
            # get the features per all words in song
            per_word_midi_features = mfe.get_per_word_features(pm_list[k], num_words, sec_per_word)
            for i in range(1, num_words):
                passed_max = 1 if i >= self.max_sequence_len else 0
                # word sequence
                seq = lyric[passed_max * (i - self.max_sequence_len + 1):i + 1]
                train_sequences.append(seq)
                # midi sequence generate
                midi_seq = per_word_midi_features[passed_max * (i - self.max_sequence_len + 1):i + 1]
                # pad midi_seq
                midi_seq = self.pad_midi_seq(midi_seq, self.max_sequence_len)
                full_midi_sequence.append(midi_seq)
                # basic features needs to be duplicated per all sequences in k song
                midi_sequence.append(midi_features[k, :])
        # pad sequences
        train_sequences_pad = np.array(pad_sequences(train_sequences, maxlen=self.max_sequence_len, padding='pre'))
        full_midi_sequence = np.stack(full_midi_sequence)
        full_midi_sequence = full_midi_sequence[:, :-1, :]  # remove last words from sequence
        sequences, label = train_sequences_pad[:, :-1], train_sequences_pad[:, -1]
        return sequences, full_midi_sequence, midi_sequence, label

    def prep_embedding_matrix(self, w2v_model, embed_dim = 300):
        # use wrod2vec pretrained embeddings to create the embedding matrix for our vocabulary
        self.num_words = len(self.tokenizer.word_index) + 2  # leave 2 words for padding and UNK
        missing_words = []

        # Prepare embedding matrix
        embed_matrix = np.zeros((self.num_words, embed_dim))
        for word, i in self.tokenizer.word_index.items():
            if word in w2v_model:
                embed_matrix[i] = w2v_model[word]
            else:
                missing_words.append(word)
        print(f'Didnt find {len(missing_words)} words in w2v model')
        return embed_matrix

    def get_num_words(self):
        return self.num_words

    def generate_lyrics(self, seed_text, num_next_words, model, midi_sequence=None, midi_sequence_full=None):
        lyrics = seed_text
        for i in range(num_next_words):
            tokens = self.tokenizer.texts_to_sequences([lyrics])[0]
            tokens_pad = pad_sequences([tokens], maxlen=self.max_sequence_len - 1, padding='pre')
            # run through midi and pad if needed
            passed_max = 1 if i >= (self.max_sequence_len-1) else 0
            if passed_max:
                #no need to pad
                midi_sequence_full_pad = midi_sequence_full[(i - (self.max_sequence_len - 2)):i + 1, :]
            else:
                midi_sequence_full_pad = midi_sequence_full[0:i+1,:]
                midi_sequence_full_pad = self.pad_midi_seq(midi_sequence_full_pad, self.max_sequence_len - 1)
            if midi_sequence_full is not None:
                prediction = model.predict([tokens_pad, midi_sequence, midi_sequence_full_pad])
            elif midi_sequence is not None:
                prediction = model.predict([tokens_pad, midi_sequence])
            else:
                prediction = model.predict(tokens_pad)
            index_predicted = np.random.choice(len(prediction), p=prediction)
            output_word = self.tokenizer.sequences_to_texts([[index_predicted]])
            lyrics += " " + output_word[0]
        return lyrics

    def pad_midi_seq(self, seq_to_pad, max_len):
        len_vector = seq_to_pad.shape[1]
        needed_pad = max_len - seq_to_pad.shape[0]
        extra_pad = np.zeros(shape=(needed_pad, len_vector))
        return np.concatenate((extra_pad, seq_to_pad))