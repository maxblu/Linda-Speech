
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
    lmfcc_feat = logfbank(audio_data,sample_rate,nfilt=80,nfft=1200)  
    # print(lmfcc_feat.shape[0]) 
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
        with open(filename, 'r',encoding='utf-8') as file:
            for line in file:
                parts = line.split()
                try:
                    audio_file = parts[0]
                    
                except Exception as identifier:
                    continue
                
                file_path = os.path.join(os.path.dirname(filename),
                                         audio_file+'.wav')                         
                try:
                    audio, sample_rate = sf.read(file_path)
                except Exception as identifier:
                    continue        
                
                
                feats[file_path] = compute_mfcc_logFilterBanks(audio, sample_rate)
                utt_len[file_path] = feats[file_path].shape[0]
                target = ' '.join(parts[1:])
                try:
                    transcripts[file_path] = [CHAR_TO_IX[i] for i in target]
                    
                except Exception as identifier:
                    # print("\n\n\n\n\n")
                    print(identifier)
                    # print("\n\n\n\n\n")


    return feats, transcripts, utt_len

def create_records():
    """ Pre-processes the raw audio and generates TFRecords.
    This function computes the mfcc features, encodes string transcripts
    into integers, and generates sequence examples for each utterance.
    Multiple sequence records are then written into TFRecord files.
    """
    for partition in sorted(glob.glob(AUDIO_PATH+'/*')):
        print('Processing' + partition)
        feats, transcripts, utt_len = process_data(partition)
        sorted_utts = sorted(utt_len, key=utt_len.get )
        # bin into groups of 100 frames.
        max_t = int(utt_len[sorted_utts[-1]]/100)
        min_t = int(utt_len[sorted_utts[0]]/100)

        # Create destination directory
        write_dir = 'processed/' + partition.split('\\')[-1]
        if tf.io.gfile.exists(write_dir):
            tf.io.gfile.remove (write_dir)
        tf.io.gfile.makedirs(write_dir)

        if os.path.basename(partition) == 'train':
            # Create multiple TFRecords based on utterance length for training
            writer = {}
            count = {}
            print('Processing training files...')
            for i in range(min_t, max_t+1):
                filename = os.path.join(write_dir, 'train' + '_' + str(i) +
                                        '.tfrecords')
                writer[i] = tf.io.TFRecordWriter(filename)
                count[i] = 0

            for utt in tqdm(sorted_utts):
                example = make_example(utt_len[utt], feats[utt].tolist(),
                                       transcripts[utt])
                index = int(utt_len[utt]/100)
                writer[index].write(example)
                count[index] += 1

            for i in range(min_t, max_t+1):
                writer[i].close()
            # print(count)

            # Remove bins which have fewer than 20 utterances
            for i in range(min_t, max_t+1):
                if count[i] < 20:
                    os.remove(os.path.join(write_dir, 'train' +
                                           '_' + str(i) + '.tfrecords'))
        else:
            # Create single TFRecord for dev and test partition
            filename = os.path.join(write_dir, os.path.basename(write_dir) +
                                    '.tfrecords')
            print('Creating', filename)
            record_writer =tf.io.TFRecordWriter(filename)
            for utt in tqdm(sorted_utts):
                example = make_example(utt_len[utt], feats[utt].tolist(),
                                       transcripts[utt])
                record_writer.write(example)
            record_writer.close()
            print('Processed '+str(len(sorted_utts))+' audio files')



AUDIO_PATH = 'data/'
ALPHABET = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZabcdefghijklmnñopqrstuvwxyz1234567890áéíóúü "
CHAR_TO_IX = {ch: i for (i, ch) in enumerate(ALPHABET)}


if __name__ == "__main__":
    # audio, sample_rate = sf.read('data/test/speaker1/1.wav')
    create_records()