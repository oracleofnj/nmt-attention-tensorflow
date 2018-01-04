# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import numpy as np
import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def _train_val_test_split(arrays, train_size, val_size):
    return [
        [
            a[:train_size],
            a[train_size:(train_size+val_size)],
            a[(train_size+val_size):]
        ]
        for a in arrays
    ]


def _read_lines(filename):
    with open(filename) as f:
        return ['<bos> ' + line.strip() + ' <eos>' for line in f]


def _build_vocab_from_sentences(sentences, min_count=5):
    all_words = [word for sentence in sentences for word in sentence.split()]
    counter = collections.Counter(all_words)
    count_pairs = sorted(
        [item for item in counter.items() if item[1] >= min_count],
        key=lambda x: (-x[1], x[0])
    )
    words = ['<unk>'] + [count_pair[0] for count_pair in count_pairs]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def _convert_sentences_to_ids(sentences, word_to_id):
    converted_sentences = [
        [
            word_to_id[word] if word in word_to_id else 0
            for word in sentence.split()
        ]
        for sentence in sentences
    ]
    return converted_sentences


def _convert_to_numpy_by_length(lang1_sentences, lang2_sentences):
    length_dict = {}
    for i, sentence in enumerate(lang1_sentences):
        sentence_length_lang1 = len(sentence)
        sentence_length_lang2 = len(lang2_sentences[i])
        if sentence_length_lang1 not in length_dict:
            length_dict[sentence_length_lang1] = [[], 0]
        length_dict[sentence_length_lang1][0].append(i)
        if sentence_length_lang2 > length_dict[sentence_length_lang1][1]:
            length_dict[sentence_length_lang1][1] = sentence_length_lang2

    input_arrays, output_arrays = [], []
    for input_len, (sentence_ids, max_output_len) in length_dict.items():
        input_arrays.append(np.array([
            lang1_sentences[i] for i in sentence_ids
        ], dtype=np.int32))
        output_arrays.append(
            np.zeros((len(sentence_ids), max_output_len), dtype=np.int32)
        )
        for output_array, sentence_id in zip(output_arrays[-1], sentence_ids):
            actual_output_len = len(lang2_sentences[sentence_id])
            output_array[:actual_output_len] = lang2_sentences[sentence_id]

    return input_arrays, output_arrays


def _convert_to_numpy(sentences):
    max_len = np.max([len(s) for s in sentences])
    output_arrays = np.zeros(
        (len(sentences), max_len), dtype=np.int32
    )
    for s, output_array in zip(sentences, output_arrays):
        sentence_length = len(s)
        output_array[:sentence_length] = s
    return output_arrays


def europarl_raw_data(
    data_path='bigdata/training',
    lang1='de-en-english.txt',
    lang2='de-en-german.txt',
    train_size=1600000,
    val_size=150000,
):
    """Load raw data from data directory "data_path".

    The dataset is from http://www.statmt.org/wmt16/translation-task.html.
    """
    lang1_path = os.path.join(data_path, lang1)
    lang2_path = os.path.join(data_path, lang2)

    split_data = _train_val_test_split(
        [_read_lines(lang1_path), _read_lines(lang2_path)],
        train_size, val_size
    )
    lang1_train, lang1_val, lang1_test = split_data[0]
    lang2_train, lang2_val, lang2_test = split_data[1]
    lang1_idx2word, lang1_word2idx = _build_vocab_from_sentences(lang1_train)
    lang2_idx2word, lang2_word2idx = _build_vocab_from_sentences(lang2_train)
    lang1_train_vectorized = _convert_sentences_to_ids(
        lang1_train,
        lang1_word2idx
    )
    lang1_val_vectorized = _convert_sentences_to_ids(
        lang1_val,
        lang1_word2idx
    )
    lang1_test_vectorized = _convert_sentences_to_ids(
        lang1_test,
        lang1_word2idx
    )
    lang2_train_vectorized = _convert_sentences_to_ids(
        lang2_train,
        lang2_word2idx
    )
    X_train, y_train = _convert_to_numpy_by_length(
        lang1_train_vectorized,
        lang2_train_vectorized
    )
    X_val = _convert_to_numpy(lang1_val_vectorized)
    X_test = _convert_to_numpy(lang1_test_vectorized)
    return {
        'vocab': {
            'lang1_idx2word': lang1_idx2word,
            'lang1_word2idx': lang1_word2idx,
            'lang2_idx2word': lang2_idx2word,
            'lang2_word2idx': lang2_word2idx,
        },
        'train': {
            'X': X_train,
            'y': y_train,
        },
        'val': {
            'X': X_val,
            'y': lang2_val,
        },
        'test': {
            'X': X_test,
            'y': lang2_test,
        },
    }


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y
