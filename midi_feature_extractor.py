import numpy as np

def get_10_part_chroma_or_pianroll(pm, fs, chroma=False):
    '''
    few midi files dont convert to 10 parts by using the fs trick
    so need to try diff fs until getting to 10 exact parts
    :param pm: pretty midi object
    :param fs: initial fs used to start
    :param chroma: if False, get_piano_roll is returned, otherwise
    :return: piano roll or chroma matrix split into 10 equal parts, np.nan if didn't succeed
    '''
    for i in range(20, 0, -1):
        if chroma:
            res = pm.get_chroma(fs=fs + fs / i)
        else:
            res = pm.get_piano_roll(fs=fs + fs / i)
        if res.shape[1] == 10:
            return res
    return np.nan


def get_midi_flat_features(pm):
    '''
    returns features based on received pretty midi object
    These features are flat - one vector is returned covering the entire meloty
    :param pm: pretty midi object
    :return: vector of features
    '''
    features = []
    # beats per second
    bps = len(pm.get_beats()) / pm.get_end_time()
    features.append(bps)
    # num of drum instruments
    features.append(sum([x.is_drum for x in pm.instruments]))
    # create a one hot vector for list of instruments
    instruments = np.zeros(128)  # 128 possible instruments
    programs = [i.program for i in pm.instruments]
    instruments[programs] = 1
    features.extend(instruments)

    fs = 1 / (pm.get_end_time() / 10)
    # piano roll features - song divided to 10 equal parts and averaged
    pr = pm.get_piano_roll(fs=fs)
    if pr.shape[1] != 10:
        # some songs need higher fs to generate 10 equal parts
        pr = get_10_part_chroma_or_pianroll(pm, fs)
    pr = pr.mean(axis=0)
    features.extend(pr)
    # chroma vector - song divided to 10 equal parts
    chroma = pm.get_chroma(fs=fs)
    if chroma.shape[1] != 10:
        # some songs need higher fs to generate 10 equal parts
        chroma = get_10_part_chroma_or_pianroll(pm, fs, chroma=True)
    chroma = chroma.reshape(chroma.size)
    features.extend(chroma)
    return np.asarray(features)

def check_overlap(note_start, note_end, word_start, word_end):
    if (((note_start <= word_start) and (note_end > word_start))
        or
       ((note_start <= word_end) and (note_start > word_start))):
        return True
    return False

def get_per_word_features(pm, num_words, sec_per_word):
    # for each word create its set of features based on sec_per_word
    result = np.zeros(shape=(num_words, 128*2))
    for i in range(0, num_words):
        start_time = sec_per_word*i + 10 #add 10 to account for intro
        end_time = start_time + sec_per_word
        instrument_vector = np.zeros(128)
        notes_vector = np.zeros(128)
        inst_cnt = np.zeros(128)
        note_cnt = np.zeros(128)
        for instrument in pm.instruments:
            for note in instrument.notes:
                if check_overlap(note.start, note.end, start_time, end_time):
                    # this note in this instruments happens in current word
                    instrument_vector[instrument.program] += note.pitch
                    inst_cnt[instrument.program] +=1
                    notes_vector[note.pitch] += note.velocity
                    note_cnt[note.pitch] += 1
                elif note.start > start_time:
                    # following notes happen later so we can break
                    break
        # avg both arrays and concatenate
        instrument_vector = np.divide(instrument_vector, inst_cnt, out=np.zeros_like(instrument_vector), where=inst_cnt!=0)
        notes_vector = np.divide(notes_vector, note_cnt, out=np.zeros_like(notes_vector), where=note_cnt!=0)
        result[i] = np.concatenate((instrument_vector,notes_vector))
    return result

