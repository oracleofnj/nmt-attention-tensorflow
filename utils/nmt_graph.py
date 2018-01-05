"""NMT without attention."""
import tensorflow as tf
import numpy as np
from .unrolled_rnn import gru_update


class NMTModel(object):
    """Holds the variables and produces the graphs."""

    def __init__(
        self,
        source_language_vocab,
        target_language_vocab,
        embedding_size,
        hidden_size,
        initializer_scale=0.1,
    ):
        """Initialize the class."""
        self.source_idx2word = source_language_vocab
        self.target_idx2word = target_language_vocab
        self.source_word2idx = dict(zip(
            source_language_vocab,
            range(len(source_language_vocab))
        ))
        self.target_word2idx = dict(zip(
            target_language_vocab,
            range(len(target_language_vocab))
        ))
        self.source_vocab_size = len(source_language_vocab)
        self.target_vocab_size = len(target_language_vocab)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        with tf.variable_scope(
            'NMTParams',
            reuse=False,
            initializer=tf.random_uniform_initializer(
                -initializer_scale, initializer_scale
            ),
        ):
            self.source_embedding_matrix = tf.get_variable(
                'source_embedding',
                [self.source_vocab_size, self.embedding_size],
                dtype=tf.float32,
            )
            self.target_embedding_matrix = tf.get_variable(
                'target_embedding',
                [self.target_vocab_size, self.embedding_size],
                dtype=tf.float32,
            )
            self.source_gru_params = {
                'U_z': tf.get_variable(
                    'U_z_source',
                    [hidden_size, hidden_size],
                    initializer=tf.orthogonal_initializer(
                        gain=initializer_scale,
                    ),
                    dtype=tf.float32,
                ),
                'W_z': tf.get_variable(
                    'W_z_source',
                    [embedding_size, hidden_size],
                    initializer=tf.orthogonal_initializer(
                        gain=initializer_scale,
                    ),
                    dtype=tf.float32,
                ),
                'b_z': tf.get_variable(
                    'b_z_source',
                    [hidden_size],
                    dtype=tf.float32,
                ),
                'U_r': tf.get_variable(
                    'U_r_source',
                    [hidden_size, hidden_size],
                    initializer=tf.orthogonal_initializer(
                        gain=initializer_scale,
                    ),
                    dtype=tf.float32,
                ),
                'W_r': tf.get_variable(
                    'W_r_source',
                    [embedding_size, hidden_size],
                    initializer=tf.orthogonal_initializer(
                        gain=initializer_scale,
                    ),
                    dtype=tf.float32,
                ),
                'b_r': tf.get_variable(
                    'b_r_source',
                    [hidden_size],
                    dtype=tf.float32,
                ),
                'U_h': tf.get_variable(
                    'U_h_source',
                    [hidden_size, hidden_size],
                    initializer=tf.orthogonal_initializer(
                        gain=initializer_scale,
                    ),
                    dtype=tf.float32,
                ),
                'W_h': tf.get_variable(
                    'W_h_source',
                    [embedding_size, hidden_size],
                    initializer=tf.orthogonal_initializer(
                        gain=initializer_scale,
                    ),
                    dtype=tf.float32,
                ),
                'b_h': tf.get_variable(
                    'b_h_source',
                    [hidden_size],
                    dtype=tf.float32,
                ),
            }
            self.target_gru_params = {
                'U_z': tf.get_variable(
                    'U_z_target',
                    [hidden_size, hidden_size],
                    initializer=tf.orthogonal_initializer(
                        gain=initializer_scale,
                    ),
                    dtype=tf.float32,
                ),
                'W_z': tf.get_variable(
                    'W_z_target',
                    [embedding_size, hidden_size],
                    initializer=tf.orthogonal_initializer(
                        gain=initializer_scale,
                    ),
                    dtype=tf.float32,
                ),
                'b_z': tf.get_variable(
                    'b_z_target',
                    [hidden_size],
                    dtype=tf.float32,
                ),
                'U_r': tf.get_variable(
                    'U_r_target',
                    [hidden_size, hidden_size],
                    initializer=tf.orthogonal_initializer(
                        gain=initializer_scale,
                    ),
                    dtype=tf.float32,
                ),
                'W_r': tf.get_variable(
                    'W_r_target',
                    [embedding_size, hidden_size],
                    initializer=tf.orthogonal_initializer(
                        gain=initializer_scale,
                    ),
                    dtype=tf.float32,
                ),
                'b_r': tf.get_variable(
                    'b_r_target',
                    [hidden_size],
                    dtype=tf.float32,
                ),
                'U_h': tf.get_variable(
                    'U_h_target',
                    [hidden_size, hidden_size],
                    initializer=tf.orthogonal_initializer(
                        gain=initializer_scale,
                    ),
                    dtype=tf.float32,
                ),
                'W_h': tf.get_variable(
                    'W_h_target',
                    [embedding_size, hidden_size],
                    initializer=tf.orthogonal_initializer(
                        gain=initializer_scale,
                    ),
                    dtype=tf.float32,
                ),
                'b_h': tf.get_variable(
                    'b_h_target',
                    [hidden_size],
                    dtype=tf.float32,
                ),
            }
            self.softmax_params = {
                'W': tf.get_variable(
                    'softmax_w',
                    [hidden_size, embedding_size],
                    dtype=tf.float32,
                ),
                'b': tf.get_variable(
                    'softmax_b',
                    [embedding_size],
                    dtype=tf.float32,
                )
            }

    def make_training_graph(
        self,
        batch_size,
        source_length,
        target_length,
    ):
        """Make all the placeholders, outputs, and training ops."""
        with tf.name_scope('placeholders_len{0}'.format(source_length)):
            inputs = tf.placeholder(
                dtype=tf.int32,
                shape=[batch_size, source_length],
                name='inputs',
            )
            targets = tf.placeholder(
                dtype=tf.int32,
                shape=[batch_size, target_length],
                name='targets',
            )
            learning_rate = tf.placeholder(
                dtype=tf.float32,
                shape=[],
                name='learning_rate',
            )
            max_norm = tf.placeholder(
                dtype=tf.float32,
                shape=[],
                name='max_norm',
            )

        with tf.name_scope('encoder_len{0}'.format(source_length)):
            pass

        with tf.name_scope('decoder_len{0}'.format(source_length)):
            pass
