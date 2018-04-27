import numpy as np
import logging
from .iterators import FileIterator, DirIterator, DirIterator, FileLineIterator, \
    TokenizedSequenceIterator, TokenIterator, IteratorChain, \
    SlidingWindowIterator
from .tokenization import DEFAULT_TOKENIZER, DEFAULT_SENT_TOKENIZER


logger = logging.getLogger(__name__)


class CorpusBuilder(object):
    def __init__(self):
        self.result = None

    def from_file(self, path):
        assert self.result is None, 'from_file or from_dir must be the first modifier'
        self.result = FileIterator(path)
        return self

    def from_dir(self, path):
        assert self.result is None, 'from_file or from_dir must be the first modifier'
        self.result = DirIterator(path)
        return self

    def by_line(self):
        self.result = FileLineIterator(self.result)
        return self

    def split(self, tokenizer=DEFAULT_TOKENIZER):
        self.result = TokenizedSequenceIterator(self.result, tokenizer=tokenizer)
        return self

    def to_tokens(self):
        self.result = TokenIterator(self.result)
        return self

    def sliding_window(self, left_ctx_size=2, right_ctx_size=2):
        self.result = SlidingWindowIterator(self.result,
                                            left_ctx_size=left_ctx_size,
                                            right_ctx_size=right_ctx_size)
        return self

    def append(self, other):
        self.result = corpus_chain(self.result, other)
        return self

    def build(self):
        # validate iterators compatibility if we feel paranoid
        assert self.result is not None
        try:
            next(iter(self.result))
        except Exception as ex:
            raise Exception('You have built invalid corpus reader!', ex)
        return self.result


# this is compact enough to not have shortcuts at all
example_file_token_corpus = CorpusBuilder().from_file('12312312').by_line().split().to_tokens().build()
example_file_sentence_corpus = CorpusBuilder().from_file('12312312').by_line().split().build()
example_dir_sliding_window_corpus = CorpusBuilder().from_dir('12312312').by_line().split().to_tokens().sliding_window().build()


# ********************** old code *******************************


def FileSentenceCorpus(path, tokenizer=DEFAULT_SENT_TOKENIZER, verbose=0):
    """
    Reads text from `path` line-by-line, splits each line into sentences, tokenizes each sentence.
    Yields data sentence-by-sentence.
    :param path: text file to read (can be archived)
    :param tokenizer: tokenizer to use to split into sentences and tokens
    :param verbose: whether to enable progressbar or not
    :return:
    """
    return CorpusBuilder().from_file(path).
    return TokenizedSequenceIterator(
        FileLineIterator(
            FileIterator(path, verbose=verbose)),
        tokenizer=tokenizer)


def DirSentenceCorpus(path, tokenizer=DEFAULT_SENT_TOKENIZER, verbose=0):
    """
    Reads text from all files from all subfolders of `path` line-by-line,
    splits each line into sentences, tokenizes each sentence.
    Yields data sentence-by-sentence.
    :param path: root directory with text files
    :param tokenizer: tokenizer to use to split into sentences and tokens
    :param verbose: whether to enable progressbar or not
    :return:
    """
    return TokenizedSequenceIterator(
        FileLineIterator(
            DirIterator(path, verbose=verbose)),
        tokenizer=tokenizer)


def DirTokenCorpus(path, tokenizer=DEFAULT_TOKENIZER, verbose=0):
    """
    Reads text from all files from all subfolders of `path` line-by-line, splits each line into tokens.
    Yields data token-by-token.
    :param path: text file to read (can be archived)
    :param tokenizer: tokenizer to use to split into sentences and tokens
    :param verbose: whether to enable progressbar or not
    :return:
    """
    return TokenIterator(
        TokenizedSequenceIterator(
            FileLineIterator(
                DirIterator(path, verbose=verbose)),
            tokenizer=tokenizer))


def FileSlidingWindowCorpus(path, left_ctx_size=2, right_ctx_size=2, tokenizer=DEFAULT_TOKENIZER, verbose=0):
    """
    Reads text from `path` line-by-line, splits each line into tokens and/or sentences (depending on tokenizer),
    and yields training samples for prediction-based distributional semantic models (like Word2Vec etc).
    Example of one yielded value: {'current': 'long', 'context': ['family', 'dashwood', 'settled', 'sussex']}
    :param path: text file to read (can be archived)
    :param tokenizer: tokenizer to use to split into sentences and tokens
    :param verbose: whether to enable progressbar or not
    :return:
    """
    return SlidingWindowIterator(
        TokenizedSequenceIterator(
            FileLineIterator(
                FileIterator(path, verbose=verbose)),
            tokenizer=tokenizer),
        left_ctx_size=left_ctx_size,
        right_ctx_size=right_ctx_size)


def DirSlidingWindowCorpus(path, left_ctx_size=2, right_ctx_size=2, tokenizer=DEFAULT_TOKENIZER, verbose=0):
    """
    Reads text from all files from all subfolders of `path` line-by-line,
    splits each line into tokens and/or sentences (depending on `tokenizer`),
    and yields training samples for prediction-based distributional semantic models (like Word2Vec etc).
    Example of one yielded value: {'current': 'long', 'context': ['family', 'dashwood', 'settled', 'sussex']}
    :param path: text file to read (can be archived)
    :param tokenizer: tokenizer to use to split into sentences and tokens
    :param verbose: whether to enable progressbar or not
    :return:
    """
    return SlidingWindowIterator(
        TokenizedSequenceIterator(
            FileLineIterator(
                DirIterator(path, verbose=verbose)),
            tokenizer=tokenizer),
        left_ctx_size=left_ctx_size,
        right_ctx_size=right_ctx_size)


def corpus_chain(*corpuses):
    """
    Join all copuses into a big single one. Like `itertools.chain`, but with proper metadata handling.
    :param corpuses: other corpuses or iterators
    :return:
    """
    return IteratorChain(corpuses)


def load_file_as_ids(path, vocabulary, tokenizer=DEFAULT_TOKENIZER):
    # use proper tokenizer from cooc
    # options to ignore sentence bounbdaries
    # specify what to do with missing words
    # replace numbers with special tokens
    result = []
    ti = FileCorpus(path).get_token_iterator(tokenizer=tokenizer)
    for token in ti:
        w = token    # specify what to do with missing words
        result.append(vocabulary.get_id(w))
    return np.array(result, dtype=np.int32)
