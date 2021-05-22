import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

class LyricsHandler():
    def __init__(self, lyrics_list, max_sequence_len = 20):
        # init tokenizer based on input lyrics
        max_sequence_len = 20
        tokenizer = Tokenizer(filters='!"#$%()*+,-./:;<=>?@[\\]^_`{|}~\t\n')  # removed &, so it's kept as a valid token
        tokenizer.fit_on_texts(lyrics_list)
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len

    def generate_train_examples(self, songs_lyrics):
        tokened_lyrics = self.tokenizer.texts_to_sequences(songs_lyrics)
        train_sequences = []
        for lyric in tokened_lyrics:
            for i in range(1, len(lyric)):
                passed_max = 1 if i >= self.max_sequence_len else 0
                seq = lyric[passed_max*(i-self.max_sequence_len+1):i+1]
                train_sequences.append(seq)
        #pad sequences
        train_sequences_pad = np.array(pad_sequences(train_sequences,maxlen=self.max_sequence_len, padding='pre'))
        sequences, label = train_sequences_pad[:,:-1],train_sequences_pad[:,-1]
        return sequences, label

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