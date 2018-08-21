import bz2
import gzip
from collections import Counter
from pathlib import Path

from utils import consts
from .logger import log


def escape(text, char_set=consts.escape_char_set):
    for char in char_set:
        if char in consts.escape_char_map:
            text = text.replace(char, consts.escape_char_map[char])
        else:
            log.warning('escape rule for {} undefined'.format(char))
    return text


def unescape(text, char_set=consts.escape_char_set):
    for char in char_set:
        if char in consts.escape_char_map:
            text = text.replace(consts.escape_char_map[char], char)
        else:
            log.warning('unescape rule for {} undefined'.format(char))
    return text


def smart_file_handler(filename: Path, mod='r'):
    if mod in ['r', 'w', 'a', 'x']:
        mod += 't'
    if filename.suffix == '.bz2':
        f = bz2.open(filename, mod)
    elif filename.suffix == '.gz':
        f = gzip.open(filename, mod)
    else:
        # noinspection PyTypeChecker
        f = open(filename, mod)
    return f


def read_vocab_count(vocab_count_file: Path):
    counter = Counter()
    with smart_file_handler(vocab_count_file, 'r') as fin:
        for line in fin.readlines():
            parts = line.strip().split('\t')
            if len(parts) == 2:
                word = parts[0]
                count = int(parts[1])
                counter[word] = count
    return counter


def read_vocab_list(vocab_list_file: Path):
    vocab_list = []
    with smart_file_handler(vocab_list_file, 'r') as fin:
        for line in fin.readlines():
            line = line.strip()
            if line:
                vocab_list.append(line)
    return vocab_list


def convert_corenlp_ner_tag(tag):
    return consts.corenlp_to_valid_mapping.get(tag, '')
