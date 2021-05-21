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