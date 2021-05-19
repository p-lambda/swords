"""
Handle conjugations using pattern.en.

Tutorial: https://github.com/p-lambda/ai_thesaurus/blob/main/tutorial
"""

import logging

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from pattern.en import lemma, singularize, pluralize, comparative, superlative, tenses
from pattern.en import conjugate as automatic_conjugate


def get_logger(name, debug=False, only_message=False):
    logger = logging.getLogger(name)

    if only_message:
        formatter = logging.Formatter('%(message)s')
    else:
        formatter = logging.Formatter('%(name)15s | %(levelname)6s | %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.propagate = False

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if debug:
        logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger('lang_utils')
tokenizer = RegexpTokenizer(r'\w+')  # Alternative: nltk.word_tokenize()
lemmatizer = WordNetLemmatizer()


def get_noun_form(word):
    if singularize(word) == word:
        return 'SINGULAR'
    else:
        return 'PLURAL'


def handle_noun(word, noun_form):
    if noun_form == 'SINGULAR':
        return singularize(word)
    elif noun_form == 'PLURAL':
        return pluralize(word)


def get_adj_form(word):
    """
    Use nltk lemmatizer instead as
        comparative() and supaerlative() from pattern.en are erroneous
        (e.g. comparative('taller') = 'tallerer')
        and lemma() is erroneous
        (e.g. lemma('taller') = 'taller').
    """
    default = lemmatizer.lemmatize(word, 'a')
    if comparative(default) == word:
        return 'COMPARATIVE'
    elif superlative(default) == word:
        return 'SUPERLATIVE'
    else:
        return 'DEFAULT'


def handle_adj(word, adj_form):
    """
    Note that comparative() and supaerlative() from pattern.en are erroneous
        (e.g. comparative('sizable') = 'sizabler').
    """
    if adj_form == 'COMPARATIVE':
        return comparative(word)
    elif adj_form == 'SUPERLATIVE':
        return superlative(word)
    else:
        return word


def get_definition(word, pos=''):
    """Get all definitions of a given word, possibly restricted by POS.

    Args:
        word (str)
        pos (str): one of ['NOUN', 'VERB', 'ADJ', 'ADV']
    """
    defs = []
    for synset in wordnet.synsets(word):
        defs.append(synset.definition())
    return defs


def get_lemma(word, pos):
    """Get lemma of a given word.

    Note that pattern.en's lemma() is erroneous.
        E.g. 'laying' -> 'layer', 'colored' -> 'colore', 'focused' -> 'focuse'
    Note that lemmatizer.lemmatize(w) is also erroneous.
        E.g. 'eyeing' -> 'eyee'
    """
    # TODO Check if the lemmatized word is in the vocabulary.
    # If not, just return the original word.

    return lemma(word)


def wordnet_pos(word):
    pos_tags = []
    for synset in wordnet.synsets(word):
        pos = synset._pos
        if pos == wordnet.NOUN:
            pos_tags.append('NOUN')
        elif pos == wordnet.VERB:
            pos_tags.append('VERB')
        elif pos == wordnet.ADJ:
            pos_tags.append('ADJ')
        elif pos == wordnet.ADV:
            pos_tags.append('ADV')
    return pos_tags


def nltk_pos(sentence, word):
    """
    Note that a modified sentence with a synonym may result in wrong pos tag.
        Therefore, it is recommended to first get possible pos tags from
        wordnet_pos() and skip nltk_pos() if the former contains only one tag.

    E.g. ("I love you.", "love") -> ['VERB'] **correct**
         ("I adoration you.", "adoration") -> ['VERB'] **wrong**
    """
    logger.debug(f'[nltk_pos] sentence={sentence}')
    logger.debug(f'[nltk_pos] word={word}')

    sentence_tokens = tokenizer.tokenize(sentence)
    word_tokens = tokenizer.tokenize(word)
    pos_per_token = {token: pos for token, pos in nltk.pos_tag(sentence_tokens)}
    pos_tags = []
    for word_token in word_tokens:
        token = word_token
        if token not in pos_per_token:
            # Search the token in the sentence_tokens.
            # When word is "would" and it's part of "wouldn't",
            # it's tokenized as "wouldn" and "'t'" by tokenizer, resulting in
            # not being able to find pos by pos_per_token["would"].
            for sentence_token in sentence_tokens:
                if sentence_token.startswith(word_token):
                    token = sentence_token
                    break
        if token not in pos_per_token:
            import ipdb; ipdb.set_trace()
            continue

        pos = pos_per_token[token]
        if pos.startswith('NN'):  # NN, NNS, NNP, NNPS
            pos_tags.append('NOUN')
        elif pos.startswith('VB'):  # VB, VBD, VBG, VBN, VBP, VBZ
            pos_tags.append('VERB')
        elif pos.startswith('JJ'):  # JJ, JJR, JJS
            pos_tags.append('ADJ')
        elif pos.startswith('RB'):  # RB, RBR, RBS
            pos_tags.append('ADV')
    return pos_tags


def get_pos_tags(
    word,
    sentence='',
    start='',
    target_word='',
):
    """Get all possible part of speech tags for a given word.

    If only a word is given, return all possible tags from wordnet.

    If sentence, start index, and target_word are given, perform
        part of speech tagging with the modified sentence with the target word
        to get the more accurate tags.

    Args:
        word (str)
        sentence (str)
        start (int)
        target_word (str)

    Returns:
        pos_tags (list of str): List of potential part of speech tags
    """
    logger.debug(f'[get_pos_tags] sentence={sentence}')
    logger.debug(f'[get_pos_tags] target_word={target_word}')
    logger.debug(f'[get_pos_tags] word={word}')

    # Prioritize wordnet if it returns only one tag
    pos_tags = list(set(wordnet_pos(word)))
    if len(pos_tags) == 1:
        return pos_tags

    # Otherwise, use nltk to select the most probable tag
    if sentence and start and target_word:
        modified = sentence[:start] + word + sentence[start + len(target_word):]
        pos_tags = nltk_pos(modified, word)
    return pos_tags


def automatic_tense(word):
    """Try finding tenses automatically."""
    try:
        automatic_tenses = tenses(word)
    except Exception as e:
        logger.debug('Could not automatically identify tense for', word)
        logger.debug(e)
        raise RuntimeError()

    tense = automatic_tenses[0]
    logger.debug(f'Automatically identified tense for {word}: {tense}')
    return tense


def heuristic_tense(word):
    """In case automatic_tense() fails, try heuristics."""
    if word.endswith('ed'):
        tense = ('past', None, None, 'indicative', 'imperfective')
    elif word.endswith('ing'):
        tense = ('present', None, None, 'indicative', 'progressive')
    elif word.endswith('s'):
        tense = ('present', 3, 'singular', 'indicative', 'imperfective')
    else:
        tense = ('infinitive', None, None, None, None)

    logger.debug(f'Heuristically identified tense for {word}: {tense}')
    return tense


def get_tense(word):
    """Find tense of a given word without considering the context of the word.

    It first tries to automatically detect tense by using pattern.en.
        If it fails, it uses heuristics to find a proper tense.

    Args:
        word (str): a target word to detect tense

    Returns:
        tense (tense, person, number, mood, aspect)
    """
    try:
        tense = automatic_tense(word)  # NOTE Can be noisy
    except:
        tense = heuristic_tense(word)  # NOTE Can be noisy
    return tense


def conjugate(word, tense):
    """Conjugate a given word with a given tense.

    If the word is composed of multiple words (e.g. "put up"),
        conjugate the first word.

    Args:
        word (str)
        tense

    Returns:
        conjugated_word (str)
    """
    multiwords = word.split(' ')
    if len(multiwords) > 1:  # NOTE Can be noisy
        conjugated_word = ' '.join(
            [automatic_conjugate(multiwords[0], tense)] + multiwords[1:])
    else:
        conjugated_word = automatic_conjugate(word, tense)
    return conjugated_word
