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
