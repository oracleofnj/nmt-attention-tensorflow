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
            embedded_encoder_inputs = tf.nn.embedding_lookup(
                self.source_embedding_matrix,
                inputs,
                name='embedded_encoder_inputs',
            )
            h_start_encoder = tf.zeros(
                [batch_size, self.hidden_size],
                name='h_start_encoder',
                dtype=tf.float32,
            )
            h_prev_encoder = h_start_encoder
            h_states_encoder = []
            for i in range(source_length):
                h_states_encoder.append(gru_update(
                    embedded_encoder_inputs[:, i, :],
                    h_prev_encoder,
                    self.source_gru_params,
                    i
                ))
                h_prev_encoder = h_states_encoder[-1]

            # concatenated_states will have shape
            # (batch_size, num_steps * hidden_size)
            concatenated_states_encoder = tf.concat(
                h_states_encoder,
                axis=1,
                name='concatenated_states_encoder'
            )
            # reshaped_states (which will get used for attention)
            # will have have shape (batch_size, num_steps, hidden_size)
            reshaped_states_encoder = tf.reshape(
                concatenated_states_encoder,
                [batch_size, source_length, self.hidden_size],
                name='reshaped_states_encoder',
            )
            # final_states will have shape
            # (batch_size, hidden_size)
            final_states = h_states_encoder[-1]

        with tf.name_scope('decoder_len{0}'.format(source_length)):
            embedded_decoder_inputs = tf.nn.embedding_lookup(
                self.target_embedding_matrix,
                targets,
                name='embedded_decoder_inputs',
            )
            h_prev_decoder = final_states
            h_states_decoder = []
            for i in range(target_length - 1):
                h_states_decoder.append(gru_update(
                    embedded_decoder_inputs[:, i, :],
                    h_prev_decoder,
                    self.target_gru_params,
                    i
                ))
                h_prev_decoder = h_states_decoder[-1]

            # concatenated_states will have shape
            # (batch_size, num_steps * hidden_size)
            concatenated_states_decoder = tf.concat(
                h_states_decoder,
                axis=1,
                name='concatenated_states_decoder'
            )
            # long_and_skinny_states will have shape
            # (batch_size * num_steps, hidden_size)
            long_and_skinny_states = tf.reshape(
                concatenated_states_decoder,
                [batch_size * (target_length - 1), self.hidden_size],
                name='long_and_skinny_states',
            )
            # long_and_skinny_logits will have shape
            # (batch_size * num_steps, vocab_size)
            long_and_skinny_antiembeddings = tf.nn.xw_plus_b(
                long_and_skinny_states,
                self.softmax_params['W'],
                self.softmax_params['b'],
                name='long_and_skinny_antiembeddings',
            )
            transposed_target_embeddings = tf.transpose(
                self.target_embedding_matrix,
                [1, 0],
                'transposed_target_embeddings',
            )
            long_and_skinny_logits = tf.matmul(
                long_and_skinny_antiembeddings,
                transposed_target_embeddings,
                name='long_and_skinny_logits',
            )
            # logits will have shape
            # (batch_size, num_steps, vocab_size)
            logits = tf.reshape(
                long_and_skinny_logits,
                [batch_size, (target_length - 1), self.target_vocab_size],
                name='logits'
            )

        with tf.name_scope('summary_len{0}'.format(source_length)):
            batch_loss = tf.contrib.seq2seq.sequence_loss(
                logits=logits,
                targets=targets[:, 1:],
                weights=tf.ones_like(
                    targets[:, 1:],
                    dtype=tf.float32
                ),
                average_across_timesteps=True,
                average_across_batch=True,
                name='batch_loss',
            )
            loss = tf.reduce_sum(
                batch_loss,
                name='loss',
            )
            predictions = tf.cast(
                tf.argmax(
                    logits,
                    axis=-1,
                ),
                tf.int32,
                name='predictions',
            )
            num_correct_predictions = tf.reduce_sum(
                tf.cast(tf.equal(predictions, targets), tf.int32),
                name='num_correct_predictions',
            )

        with tf.name_scope('train_ops_len{0}'.format(source_length)):
            trainable_variables = tf.trainable_variables()
            unclipped_gradients = tf.gradients(loss, trainable_variables)
            gradient_global_norm = tf.global_norm(
                unclipped_gradients,
                name='gradient_global_norm'
            )
            clipped_gradients, _ = tf.clip_by_global_norm(
                unclipped_gradients,
                max_norm,
                name='clipped_gradients'
            )
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.apply_gradients(
                zip(clipped_gradients, trainable_variables),
            )

        return {
            'placeholders': {
                'inputs': inputs,
                'targets': targets,
                'learning_rate': learning_rate,
                'max_norm': max_norm,
            },
            'outputs': {
                'loss': loss,
                'num_correct_predictions': num_correct_predictions,
            },
            'train_ops': {
                'train_op': train_op,
                'gradient_global_norm': gradient_global_norm,
            }
        }
