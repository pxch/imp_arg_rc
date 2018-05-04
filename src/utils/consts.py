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

# type of dependency parses to use from Stanford CoreNLP tool
corenlp_dependency_type = 'enhanced-plus-plus-dependencies'

# mappings from CoreNLP NER tags to valid NER tags
corenlp_to_valid_mapping = {
    'PERSON': 'PER',
    'ORGANIZATION': 'ORG',
    'LOCATION': 'LOC',
    'MISC': '',  # 'MISC',
    'MONEY': 'NUM',
    'NUMBER': 'NUM',
    'ORDINAL': 'NUM',
    'PERCENT': 'NUM',
    'DATE': 'TEMP',
    'TIME': 'TEMP',
    'DURATION': 'TEMP',
    'SET': 'TEMP'
}

# 10 most frequent predicate from training corpus (English Wikipedia 20160901)
stop_preds = ['have', 'include', 'use', 'make', 'play',
              'take', 'win', 'give', 'serve', 'receive']

pred_count_thres = 100000
