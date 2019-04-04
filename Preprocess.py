
import os
import glob
import soundfile as sf
from python_speech_features import logfbank
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import keras_preprocessing
import keras_applications
from pprint import pprint

#done
def compute_mfcc_logFilterBanks(audio_data, sample_rate):
    ''' Computes the mel-frequency cepstral coefficients.
    The audio time series is normalised and its mfcc features are computed.

    Args:
        audio_data: time series of the speech utterance.
        sample_rate: sampling rate.
    Returns:
        mfcc_feat:[num_frames x F] matrix representing the mfcc.

    '''

    # audio_data = audio_data - np.mean(audio_data)
    # audio_data = audio_data / np.max(audio_data)
    lmfcc_feat = logfbank(audio_data,sample_rate,nfilt=80)   
    return lmfcc_feat

#done
def make_example(seq_len, spec_feat, labels):
    ''' Creates a SequenceExample for a single utterance.
    This function makes a SequenceExample given the sequence length,
    mfcc features and corresponding transcript.
    These sequence examples are read using tf.parse_single_sequence_example
    during training.

    Note: Some of the tf modules used in this function(such as
    tf.train.Feature) do not have comprehensive documentation in v0.12.
    This function was put together using the test routines in the
    tensorflow repo.
    See: https://github.com/tensorflow/tensorflow/
    blob/246a3724f5406b357aefcad561407720f5ccb5dc/
    tensorflow/python/kernel_tests/parsing_ops_test.py


    Args:
        seq_len: integer represents the sequence length in time frames.
        spec_feat: [TxF] matrix of mfcc features.
        labels: list of ints representing the encoded transcript.
    Returns:
        Serialized sequence example.

    '''
    # Feature lists for the sequential features of the example
    feats_list = [tf.train.Feature(float_list=tf.train.FloatList(value=frame))
                  for frame in spec_feat]
    feat_dict = {"feats": tf.train.FeatureList(feature=feats_list)}
    sequence_feats = tf.train.FeatureLists(feature_list=feat_dict)

    # Context features for the entire sequence
    len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len]))
    label_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=labels))

    context_feats = tf.train.Features(feature={"seq_len": len_feat,
                                               "labels": label_feat})

    ex = tf.train.SequenceExample(context=context_feats,
                                  feature_lists=sequence_feats)

    return ex.SerializeToString()


def process_data(partition):
    """ Reads audio waveform and transcripts from a dataset partition
    and generates mfcc featues.

    Args:
        parition - represents the dataset partition name.

    Returns:
        feats: dict containing mfcc feature per utterance
        transcripts: dict of lists representing transcript.
        utt_len: dict of ints holding sequence length of each
                 utterance in time frames.

    """

    feats = {}
    transcripts = {}
    utt_len = {}  # Required for sorting the utterances based on length

    for filename in glob.iglob(partition+'/**/*.txt', recursive=True):
        with open(filename, 'r') as file:
            for line in file:
                parts = line.split()
                audio_file = parts[0]
                file_path = os.path.join(os.path.dirname(filename),
                                         audio_file+'.wav')
                audio, sample_rate = sf.read(file_path)
                feats[audio_file] = compute_mfcc_logFilterBanks(audio, sample_rate)
                utt_len[audio_file] = feats[audio_file].shape[0]
                target = ' '.join(parts[1:])
                # transcripts[audio_file] = [CHAR_TO_IX[i] for i in target]
    return feats, transcripts, utt_len




if __name__ == "__main__":

    audio, sample_rate = sf.read('s11.wav')

    matrix_coef=compute_mfcc_logFilterBanks(audio,sample_rate)

    print(np.shape(matrix_coef))
     
    for i in matrix_coef:
        pprint(i)

    # print(matrix_coef)
