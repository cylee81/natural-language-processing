"""Dataset loader and data utilities.

Author:
    Shrey Desai and Yasumasa Onoe
"""
import spacy
import collections
import itertools
import torch
from tqdm import tqdm
import time, pickle, os

from torch.utils.data import Dataset
from random import shuffle
from utils import cuda, load_dataset

from joblib import Parallel, delayed
import spacy
nlp = spacy.load("en_core_web_sm")

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'

nlp = spacy.load("en_core_web_sm")
stop_word = stopwords.words()

def _get(idx, samples, passage_ner_types, stopwordss, token):

    # nltk.download('stopwords')
    PAD_TOKEN = '[PAD]'
    UNK_TOKEN = '[UNK]'

    NER_LABELS = [
    'CARDINAL',
    'DATE',
    'EVENT',
    'FAC',
    'GPE',
    'LANGUAGE',
    'LAW',
    'LOC',
    'MONEY',
    'NORP',
    'ORDINAL',
    'ORG',
    'PERCENT',
    'PERSON',
    'PRODUCT',
    'QUANTITY',
    'TIME',
    'WORK_OF_ART'
    ]

    NER_MAPPINGS = {NER_LABELS[idx]: idx for idx in range(len(NER_LABELS))}

    def map_ner_label_to_idx(ner_label):
        if ner_label in NER_MAPPINGS:
            return NER_MAPPINGS[ner_label]
        return len(NER_LABELS)

    qid, passage, question, answer_start, answer_end = samples[idx]

    old_passage = passage
    old_question = question
    # Convert words to tensor.
    passage_ids = torch.tensor(
        token.convert_tokens_to_ids(passage)
    )
    # print("convert2")
    question_ids = torch.tensor(
        token.convert_tokens_to_ids(question)
    )
    answer_start_ids = torch.tensor(answer_start)
    answer_end_ids = torch.tensor(answer_end)

    answer = passage[answer_start:answer_end+1]
    ner_type = None

    filtered_answer_lst = []
    for word in answer:
        if not len(word) > 0: continue
        if word in stopwordss: continue
        filtered_answer_lst.append(word)

    new_passage = []

    start_sent = 0
    contains_entity = False

    t1 = time.time()

    for idx,w in enumerate(passage):
        is_entity = False
        for ent in passage_ner_types.ents:
            if ent.start > idx:
                break
            if ent.start <= idx and ent.end > idx :
                is_entity = True
                break

        contains_entity = contains_entity or is_entity
        if w == '.':
            if contains_entity:
                new_passage.extend(passage[start_sent:idx + 1])
            start_sent = idx + 1
            contains_entity = False

    passage = new_passage
    for ent in passage_ner_types.ents:
        if all([w in ent.text for w in filtered_answer_lst]):
            ner_type = ent.label_
            break
    if ner_type is None:
        ner_type = torch.tensor([len(NER_LABELS)])
    else:
        ner_type = map_ner_label_to_idx(ner_type)
        ner_type = torch.tensor(ner_type)

    return passage_ids, question_ids, answer_start_ids, answer_end_ids, ner_type, old_passage, old_question

NER_LABELS = [
    'CARDINAL',
    'DATE',
    'EVENT',
    'FAC',
    'GPE',
    'LANGUAGE',
    'LAW',
    'LOC',
    'MONEY',
    'NORP',
    'ORDINAL',
    'ORG',
    'PERCENT',
    'PERSON',
    'PRODUCT',
    'QUANTITY',
    'TIME',
    'WORK_OF_ART'
]

NER_MAPPINGS = {NER_LABELS[idx]: idx for idx in range(len(NER_LABELS))}

def map_ner_label_to_idx(ner_label):
    if ner_label in NER_MAPPINGS:
        return NER_MAPPINGS[ner_label]
    return len(NER_LABELS)

class Vocabulary:
    """
    This class creates two dictionaries mapping:
        1) words --> indices,
        2) indices --> words.

    Args:
        samples: A list of training examples stored in `QADataset.samples`.
        vocab_size: Int. The number of top words to be used.

    Attributes:
        words: A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
            All words will be lowercased.
        encoding: A dictionary mapping words (string) to indices (int).
        decoding: A dictionary mapping indices (int) to words (string).
    """
    def __init__(self, samples, vocab_size):
        self.words = self._initialize(samples, vocab_size)
        self.encoding = {word: index for (index, word) in enumerate(self.words)}
        self.decoding = {index: word for (index, word) in enumerate(self.words)}

    def _initialize(self, samples, vocab_size):
        """
        Counts and sorts all tokens in the data, then it returns a vocab
        list. `PAD_TOKEN and `UNK_TOKEN` are added at the beginning of the
        list. All words are lowercased.

        Args:
            samples: A list of training examples stored in `QADataset.samples`.
            vocab_size: Int. The number of top words to be used.

        Returns:
            A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
        """
        vocab = collections.defaultdict(int)
        for (_, passage, question, _, _) in samples:
            for token in itertools.chain(passage, question):
                vocab[token.lower()] += 1
        top_words = [
            word for (word, _) in
            sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        ][:vocab_size]
        words = [PAD_TOKEN, UNK_TOKEN] + top_words
        return words

    def __len__(self):
        return len(self.words)


class Tokenizer:
    """
    This class provides two methods converting:
        1) List of words --> List of indices,
        2) List of indices --> List of words.

    Args:
        vocabulary: An instantiated `Vocabulary` object.

    Attributes:
        vocabulary: A list of top words (string) sorted by frequency.
            `PAD_TOKEN` (at position 0) and `UNK_TOKEN` (at position 1) are
            prepended.
        pad_token_id: Index of `PAD_TOKEN` (int).
        unk_token_id: Index of `UNK_TOKEN` (int).
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.pad_token_id = self.vocabulary.encoding[PAD_TOKEN]
        self.unk_token_id = self.vocabulary.encoding[UNK_TOKEN]

    def convert_tokens_to_ids(self, tokens):
        """
        Converts words to corresponding indices.

        Args:
            tokens: A list of words (string).

        Returns:
            A list of indices (int).
        """
        return [
            self.vocabulary.encoding.get(token, self.unk_token_id)
            for token in tokens
        ]

    def convert_ids_to_tokens(self, token_ids):
        """
        Converts indices to corresponding words.

        Args:
            token_ids: A list of indices (int).

        Returns:
            A list of words (string).
        """
        return [
            self.vocabulary.decoding.get(token_id, UNK_TOKEN)
            for token_id in token_ids
        ]


class QADataset(Dataset):
    """
    This class creates a data generator.

    Args:
        args: `argparse` object.
        path: Path to a data file (.gz), e.g. "datasets/squad_dev.jsonl.gz".

    Attributes:
        args: `argparse` object.
        meta: Dataset metadata (e.g. dataset name, split).
        elems: A list of raw examples (jsonl).
        samples: A list of preprocessed examples (tuple). Passages and
            questions are shortened to max sequence length.
        tokenizer: `Tokenizer` object.
        batch_size: Int. The number of example in a mini batch.
    """
    def __init__(self, args, path):
        self.args = args
        self.meta, self.elems = load_dataset(path)
        self.samples = self._create_samples()
        self.tokenizer = None
        self.batch_size = args.batch_size if 'batch_size' in args else 1
        self.pad_token_id = self.tokenizer.pad_token_id \
            if self.tokenizer is not None else 0

    def _create_samples(self):
        """
        Formats raw examples to desired form. Any passages/questions longer
        than max sequence length will be truncated.

        Returns:
            A list of words (string).
        """
        samples = []
        for elem in self.elems:
            # Unpack the context paragraph. Shorten to max sequence length.
            passage = [
                token.lower() for (token, offset) in elem['context_tokens']
            ][:self.args.max_context_length]

            # Each passage has several questions associated with it.
            # Additionally, each question has multiple possible answer spans.
            for qa in elem['qas']:
                qid = qa['qid']
                question = [
                    token.lower() for (token, offset) in qa['question_tokens']
                ][:self.args.max_question_length]

                # Select the first answer span, which is formatted as
                # (start_position, end_position), where the end_position
                # is inclusive.
                answers = qa['detected_answers']
                answer_start, answer_end = answers[0]['token_spans'][0]
                samples.append(
                    (qid, passage, question, answer_start, answer_end)
                )
        return samples

    def _create_data_generator(self, shuffle_examples=False):
        """
        Converts preprocessed text data to Torch tensors and returns a
        generator.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A generator that iterates through all examples one by one.
            (Tuple of tensors)
        """
        if self.tokenizer is None:
            raise RuntimeError('error: no tokenizer registered')

        example_idxs = list(range(len(self.samples)))
        if shuffle_examples:
            shuffle(example_idxs)

        passages = []
        questions = []
        start_positions = []
        end_positions = []
        ner_types = []
        samples = self.samples.copy()
        raw_passages = []
        raw_answers = []

        # for idx in tqdm(example_idxs):
        raw_passage = []
        for idx in tqdm(example_idxs):
            _, passage, _, _, _ = samples[idx]
            raw_passage.append(" ".join(passage))

        if self.args.mode == "train":
            if os.path.isfile(self.args.train_pickle_path):
                print(f"Loading pickle from: {self.args.train_pickle_path}")
                with open(self.args.train_pickle_path, 'rb') as handle:
                    passage_ner_types = pickle.load(handle)
            else:
                t1 = time.time()
                passage_ner_types = list(nlp.pipe(raw_passage, n_process=-1))
                print(f"Saving pickle to: {self.args.train_pickle_path}")
                with open(self.args.train_pickle_path, 'wb') as handle:
                    pickle.dump(passage_ner_types, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"it takes {time.time()-t1}")

        if self.args.mode == "dev":
            if os.path.isfile(self.args.dev_pickle_path):
                print(f"Loading pickle from: {self.args.dev_pickle_path}")
                with open(self.args.dev_pickle_path, 'rb') as handle:
                    passage_ner_types = pickle.load(handle)
            else:
                t1 = time.time()
                passage_ner_types = list(nlp.pipe(raw_passage, n_process=-1))
                print(f"Saving pickle to: {self.args.dev_pickle_path}")
                with open(self.args.dev_pickle_path, 'wb') as handle:
                    pickle.dump(passage_ner_types, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"it takes {time.time()-t1}")

        # print(f"it takes {time.time()-t1}")
        res = Parallel(n_jobs=5, backend="threading")(delayed(_get)(i, samples, passage_ner_types[i], stop_word, self.tokenizer)for i in example_idxs)

        for re in tqdm(res):
            passages.append(re[0])
            questions.append(re[1])
            start_positions.append(re[2])
            end_positions.append(re[3])
            ner_types.append(re[4])
            raw_passages.append(re[5])
            raw_answers.append(re[6])
            # questions.append(question_ids)
            # start_positions.append(answer_start_ids)
            # end_positions.append(answer_end_ids)
            # ner_types.append(ner_type)

        return zip(passages, questions, start_positions, end_positions, ner_types, raw_passages, raw_answers)

    def _create_batches(self, generator, batch_size):
        """
        This is a generator that gives one batch at a time. Tensors are
        converted to "cuda" if necessary.

        Args:
            generator: A data generator created by `_create_data_generator`.
            batch_size: Int. The number of example in a mini batch.

        Yields:
            A dictionary of tensors containing a single batch.
        """
        current_batch = [None] * batch_size
        no_more_data = False
        ner_types = torch.zeros(batch_size)
        # Loop through all examples.
        while True:
            bsz = batch_size
            # Get examples from generator
            for i in range(batch_size):
                try:
                    current_batch[i] = list(next(generator))
                except StopIteration:  # Run out examples
                    no_more_data = True
                    bsz = i  # The size of the last batch.
                    break
            # Stop if there's no leftover examples
            if no_more_data and bsz == 0:
                break

            passages = []
            questions = []
            raw_passages = []
            raw_questions = []
            start_positions = torch.zeros(bsz)
            end_positions = torch.zeros(bsz)
            max_passage_length = 0
            max_question_length = 0
            # Check max lengths for both passages and questions
            for ii in range(bsz):
                passages.append(current_batch[ii][0])
                questions.append(current_batch[ii][1])
                start_positions[ii] = current_batch[ii][2]
                end_positions[ii] = current_batch[ii][3]
                ner_types[ii] = current_batch[ii][4]
                raw_passages.append(current_batch[ii][5])
                raw_questions.append(current_batch[ii][6])

                max_passage_length = max(
                    max_passage_length, len(current_batch[ii][0])
                )
                max_question_length = max(
                    max_question_length, len(current_batch[ii][1])
                )

            # Assume pad token index is 0. Need to change here if pad token
            # index is other than 0.
            padded_passages = torch.zeros(bsz, max_passage_length)
            padded_questions = torch.zeros(bsz, max_question_length)
            # Pad passages and questions
            for iii, passage_question in enumerate(zip(passages, questions)):
                passage, question = passage_question
                padded_passages[iii][:len(passage)] = passage
                padded_questions[iii][:len(question)] = question

            # Create an input dictionary
            batch_dict = {
                'passages': cuda(self.args, padded_passages).long(),
                'questions': cuda(self.args, padded_questions).long(),
                'start_positions': cuda(self.args, start_positions).long(),
                'end_positions': cuda(self.args, end_positions).long(),
                'ner_types': cuda(self.args, ner_types).long(),
                'raw_passages': raw_passages,
                'raw_questions': raw_questions
            }

            if no_more_data:
                if bsz > 0:
                    # This is the last batch (smaller than `batch_size`)
                    yield batch_dict
                break
            yield batch_dict

    def get_batch(self, shuffle_examples=False):
        """
        Returns a data generator that supports mini-batch.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A data generator that iterates though all batches.
        """
        return self._create_batches(
            self._create_data_generator(shuffle_examples=shuffle_examples),
            self.batch_size
        )

    def register_tokenizer(self, tokenizer):
        """
        Stores `Tokenizer` object as an instance variable.

        Args:
            tokenizer: If `True`, shuffle examples. Default: `False`
        """
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)
