from utils.char_map import char_map, index_map


def text_to_int_sequence(text):
    """ Use a character map and convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence


def int_to_text_sequence(seq):
    """ Use a index map and convert int to a text sequence """
    text_sequence = []
    for c in seq:
        if c == 35: # ctc/pad char
            ch = ''
        else:
            ch = index_map[c]
        text_sequence.append(ch)
    return text_sequence
