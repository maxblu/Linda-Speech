
# Based on tutorial: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

from random import shuffle as shuf

import numpy as np
from keras.utils import Sequence
from librosa.feature import mfcc, melspectrogram

from utils.feature_utils import load_audio, convert_and_pad_transcripts, extract_mfcc_and_pad, \
    extract_mel_spectrogram_and_pad


class DataGenerator(Sequence):
    """
    Thread safe data generator for the fit_generator

    Args:
        df (Pandas.Dataframe): dataframes containing (filename, filesize, transcript)
        batch_size (int): size of each batch
        frame_length (int): size of each frame (samples per frame)
        hop_length (int): how far to move center of each frame when splitting audio time series
        mfcc_features (int, default=26): how many mfcc-features to extract for each frame
        epoch_length (int, default=0): the number of batches in each epoch, if set to zero it uses all available data
        shuffle (boolean, default=True): whether to shuffle the indexes in each batch

    
    """

    def __init__(self, df, feature_type='mfcc', batch_size=32, frame_length=320, hop_length=160, n_mels=40,
                 mfcc_features=26, epoch_length=0, shuffle=True):
        self.df = df.copy()
        self.type = feature_type
        self.batch_size = batch_size
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.mfcc_features = mfcc_features
        self.n_mels = n_mels
        self.epoch_length = epoch_length
        self.shuffle = shuffle

        # Initializing indexes
        self.indexes = np.arange(len(self.df))

    def __len__(self):
        """Denotes the number of batches per epoch"""
        if (self.epoch_length == 0) | (self.epoch_length > int(np.floor(self.df.shape[0]/self.batch_size))):
            self.epoch_length = int(np.floor(self.df.shape[0] / self.batch_size))
        return self.epoch_length

    def __getitem__(self, batch_index):
        """
        Generates a batch of correctly shaped X and Y data

        :param batch_index: index of the batch to generate
        :return: input dictionary containing:
                'the_input':     np.ndarray[shape=(batch_size, max_seq_length, mfcc_features)]: input audio data
                'the_labels':    np.ndarray[shape=(batch_size, max_transcript_length)]: transcription data
                'input_length':  np.ndarray[shape=(batch_size, 1)]: length of each sequence (numb of frames) in x_data
                'label_length':  np.ndarray[shape=(batch_size, 1)]: length of each sequence (numb of letters) in y_data
                 output dictionary containing:
                'ctc':           np.ndarray[shape=(batch_size, 1)]: dummy data for dummy loss function

        """

        # Generate indexes of current batch
        indexes_in_batch = self.indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        # Shuffle indexes within current batch if shuffle=true
        if self.shuffle:
            shuf(indexes_in_batch)

        # Load audio and transcripts
        x_data_raw, y_data_raw, sr = load_audio(self.df, indexes_in_batch)

        # Preprocess and pad data
        x_data, input_length = self.extract_features_and_pad(x_data_raw, sr)
        y_data, label_length = convert_and_pad_transcripts(y_data_raw)

        # print ("\nx_data shape: ", x_data.shape)
        # print ("y_data shape: ", y_data.shape)
        # print ("input_length shape: ", input_length.shape)
        # print ("label_length shape: ", label_length.shape)
        # print ("input length: ", input_length)
        # print ("label_length: ", label_length, "\n")

        inputs = {'the_input': x_data,
                  'the_labels': y_data,
                  'input_length': input_length,
                  'label_length': label_length}

        outputs = {'ctc': np.zeros([self.batch_size])} # dummy data for dummy loss function

        return inputs, outputs

    def extract_features_and_pad(self, x_data_raw, sr):
        """
        Converts list of audio time series to MFCC or melspectrogram
        Zero-pads each sequence to be equal length to the longest sequence.
        Stores the length of each feature-sequence before padding for the CTC

        :param x_data_raw: list with audio time series
        :param sr: sampling rate of frames
        :return: x_data: numpy array with padded feature-sequence (MFCC or melspectrogram)
                 input_length: numpy array containing unpadded length of each feature-sequence
        """
        # print('Extracting feautures and padding')    
        # Finds longest frame in batch for padding
        max_x_length = self.get_seq_size(max(x_data_raw, key=len), sr)

        if self.type == 'mfcc':
            x_data = np.empty([0, max_x_length, self.mfcc_features])
            len_x_seq = []
            # print('extracting mfcc and pading')
            # Extract mfcc features and pad so every frame-sequence is equal max_x_length
            for i in range(0, len(x_data_raw)):
                x, x_len = extract_mfcc_and_pad(x_data_raw[i], sr, max_x_length, self.frame_length, self.hop_length,
                                                self.mfcc_features, self.n_mels)
                x_data = np.insert(x_data, i, x, axis=0)
                len_x_seq.append(x_len - 2)  # -2 because ctc discards the first two outputs of the rnn network

            # Convert input length list to numpy array
            input_length = np.array(len_x_seq)
            return x_data, input_length

        elif self.type == 'spectrogram':
            x_data = np.empty([0, max_x_length, self.n_mels])
            len_x_seq = []

            # Extract mel spectrogram features and pad so every frame-sequence is equal max_x_length
            for i in range(0, len(x_data_raw)):
                x, x_len = extract_mel_spectrogram_and_pad(x_data_raw[i], sr, max_x_length, self.frame_length,
                                                           self.hop_length, self.n_mels)
                x_data = np.insert(x_data, i, x, axis=0)
                len_x_seq.append(x_len - 2)  # -2 because ctc discards the first two outputs of the rnn network

            # Convert input length list to numpy array
            input_length = np.array(len_x_seq)
            return x_data, input_length

        else:
            raise ValueError('Not a valid feature type: ', self.type)

    def get_seq_size(self, frames, sr):
        """
        Get audio sequence size of audio time series when converted to mfcc-features or mel spectrogram

        :param frames: audio time series
        :param sr: sampling rate of frames
        :return: sequence size of mfcc-converted audio
        """

        if self.type == 'mfcc':
            mfcc_frames = mfcc(frames, sr, n_fft=self.frame_length, hop_length=self.hop_length,
                               n_mfcc=self.mfcc_features, n_mels=self.n_mels)
            return mfcc_frames.shape[1]

        elif self.type == 'spectrogram':
            spectrogram = melspectrogram(frames, sr, n_fft=self.frame_length, hop_length=self.hop_length,
                                         n_mels=self.n_mels)
            return spectrogram.shape[1]

        else:
            raise ValueError('Not a valid feature type: ', self.type)