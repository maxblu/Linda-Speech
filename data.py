import os

import pandas as pd

from utils.char_map import char_map
from utils.text_utils import text_to_int_sequence


#######################################################

def clean(word):
    new = word.lower().replace('.', '')
    new = new.replace(',', '')
    new = new.replace(';', '')
    new = new.replace('"', '')
    new = new.replace('!', '')
    new = new.replace('?', '')
    new = new.replace(':', '')
    new = new.replace('-', '')
    return new


def combine_all_wavs_and_trans_from_csvs(csvslist, sortagrad=False, createwordlist=False, delBigTranscripts=False):
    '''Assume that data is in csv already exists with data in form
        path, size, transcript
    '''

    df_all = pd.DataFrame()

    for csv in csvslist.split(','):
        print("Reading csv:",csv)

        if os.path.isfile(csv):
            try:
                df_new = pd.read_csv(csv, sep=',', encoding='ascii')
            except:
                print("NOT - ASCII, use UTF-8")
                df_new = pd.read_csv(csv, sep=',', encoding='utf-8')
               

            df_all = df_all.append(df_new)

    print("Finished reading in data")

    if delBigTranscripts:
        print("removing any sentences that are too big- tweetsize")
        df_final = df_all[df_all['transcript'].map(len) <= 280]
    else:
        df_final = df_all

    
    listcomb = df_all['transcript'].tolist()
    print("Total number of files:", len(listcomb))

    
    listcomb = df_final['transcript'].tolist()
    print("Total number of files (after reduction):", len(listcomb))

    comb = []

    
    for t in listcomb:
        

        if isinstance(t,float):
            print(listcomb.index(t) )
            continue
        comb.append(' '.join(t.split()))
    

    
    ## SIZE CHECKS
    max_intseq_length = get_max_intseq(comb)
    num_classes = get_number_of_char_classes()

    print("max_intseq_length:", max_intseq_length)
    print("numclasses:", num_classes)

    # VOCAB CHECKS
    all_words, max_trans_charlength = get_words(comb)
    print("max_trans_charlength:", max_trans_charlength)


   
    all_vocab = set(all_words)
    print("Words:", len(all_words))
    print("Vocab:", len(all_vocab))

    dataproperties = {
        'target': "librispeech",
        'num_classes': num_classes,
        'all_words': all_words,
        'all_vocab': all_vocab,
        'max_trans_charlength': max_trans_charlength,
        'max_intseq_length': max_intseq_length
    }

    if sortagrad:
        df_final = df_final.sort_values(by='filesize', ascending=True)
    else:
        df_final = df_final.sample(frac=1).reset_index(drop=True)

    #remove mem
    del df_all
    del listcomb

    return dataproperties, df_final


##DATA CHECKS RUN ALL OF THESE

def get_words(comb):
    
    max_trans_charlength = 0
    all_words = []

    for count, sent in enumerate(comb):
        # count length
        if len(sent) > max_trans_charlength:
            max_trans_charlength = len(sent)
        # build vocab
        for w in sent.split():
            all_words.append(clean(w))

    return all_words, max_trans_charlength


def get_max_intseq(comb):
    max_intseq_length = 0
    for x in comb:
        try:
            y = text_to_int_sequence(x)
            if len(y) > max_intseq_length:
                max_intseq_length = len(y)
        except:
            print("error at:", x)
    return max_intseq_length


def get_number_of_char_classes():
    
    num_classes = len(char_map)+1 ##need +1 for ctc null char +1 pad
    return num_classes


