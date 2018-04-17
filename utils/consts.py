# valid NER tags (combined from CoreNLP and Ontonotes)
valid_ner_tags = ['PER', 'ORG', 'LOC', 'TEMP', 'NUM']

# set of escape characters in constructing the word/lemma/pos of a token
escape_char_set = [' // ', '/', ';', ',', ':', '-']

# mappings from escape characters to their representations
escape_char_map = {
    ' // ': '@slashes@',
    '/': '@slash@',
    ';': '@semicolon@',
    ',': '@comma@',
    ':': '@colon@',
    '-': '@dash@',
    '_': '@underscore@'}
