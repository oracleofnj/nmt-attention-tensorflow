"""NMT without attention."""
import tensorflow as tf
from .unrolled_rnn import gru_update


class AttentionModel(object):
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
                'C_z': tf.get_variable(
                    'C_z_target',
                    [hidden_size, hidden_size],
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
                'C_r': tf.get_variable(
                    'C_r_target',
                    [hidden_size, hidden_size],
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
                'C_h': tf.get_variable(
                    'C_h_target',
                    [hidden_size, hidden_size],
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
            self.attention_params = {
                'W': tf.get_variable(
                    'attention_w',
                    [hidden_size, hidden_size],
                    dtype=tf.float32,
                )
            }

    def _attn_gru_update(
        self,
        x_t,
        c_t,
        h_t_minus_1,
        timestep=None
    ):
        U_z, W_z, C_z, b_z = \
            self.target_gru_params['U_z'], \
            self.target_gru_params['W_z'], \
            self.target_gru_params['C_z'], \
            self.target_gru_params['b_z']
        U_r, W_r, C_r, b_r = \
            self.target_gru_params['U_r'], \
            self.target_gru_params['W_r'], \
            self.target_gru_params['C_r'], \
            self.target_gru_params['b_r']
        U_h, W_h, C_h, b_h = \
            self.target_gru_params['U_h'], \
            self.target_gru_params['W_h'], \
            self.target_gru_params['C_h'], \
            self.target_gru_params['b_h']
        with tf.name_scope('gru_calculations'):
            r_t = tf.sigmoid(
                tf.matmul(x_t, W_r) +
                tf.matmul(h_t_minus_1, U_r) +
                tf.matmul(c_t, C_r) +
                b_r,
                name='r' + (
                    '_{0}'.format(timestep) if timestep is not None else ''
                ),
            )
            z_t = tf.sigmoid(
                tf.matmul(x_t, W_z) +
                tf.matmul(h_t_minus_1, U_z) +
                tf.matmul(c_t, C_z) +
                b_z,
                name='z' + (
                    '_{0}'.format(timestep) if timestep is not None else ''
                ),
            )
            h_tilde_t = tf.tanh(
                tf.matmul(x_t, W_h) +
                tf.matmul(h_t_minus_1 * r_t, U_h) +
                tf.matmul(c_t, C_h) +
                b_h,
                name='h_tilde' + (
                    '_{0}'.format(timestep) if timestep is not None else ''
                ),
            )
            h_t = (1 - z_t) * h_t_minus_1 + z_t * h_tilde_t
        return h_t

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
            # reshaped_states will have have shape
            # (batch_size, num_steps, hidden_size)
            reshaped_states_encoder = tf.reshape(
                concatenated_states_encoder,
                [batch_size, source_length, self.hidden_size],
                name='reshaped_states_encoder',
            )
            # attended_states will have shape
            # (batch_size, num_steps, hidden_size)
            attended_states = tf.identity(
                tf.einsum(
                    'ij,fgj->fgi',
                    self.attention_params['W'],
                    reshaped_states_encoder,
                ),
                name='attended_states',
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
            attention_weights = []
            h_states_decoder = []
            for i in range(target_length - 1):
                # attention_weights_unnormalized will have shape
                # (batch_size, source_length)
                attention_weights_unnormalized = tf.identity(
                    tf.einsum(
                        'ik,ijk->ij',
                        h_prev_decoder,
                        attended_states,
                    ),
                    name='attention_weights_unnormalized{0}'.format(i)
                )
                # attention_weights_normalized will have shape
                # (batch_size, source_length)
                attention_weights_normalized = tf.nn.softmax(
                    attention_weights_unnormalized,
                    name='attention_weights_normalized{0}'.format(i)
                )
                attention_weights.append(attention_weights_normalized)
                # context_vector will have shape
                # (batch_size, hidden_size)
                context_vector = tf.identity(
                    tf.einsum(
                        'ij,ijk->ik',
                        attention_weights_normalized,
                        reshaped_states_encoder,
                    ),
                    name='context_vector{0}'.format(i)
                )
                h_states_decoder.append(self._attn_gru_update(
                    embedded_decoder_inputs[:, i, :],
                    context_vector,
                    h_prev_decoder,
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
            # concatenated_attention_weights will have shape
            # (batch_size, (target_length - 1) * source_length)
            concatenated_attention_weights = tf.concat(
                attention_weights,
                axis=1,
                name='concatenated_attention_weights'
            )
            # reshaped_attention_weights will have have shape
            # (batch_size, (target_length - 1) * source_length)
            reshaped_attention_weights = tf.reshape(
                concatenated_attention_weights,
                [batch_size, target_length - 1, source_length],
                name='attention_weights',
            )

        with tf.name_scope('summary_len{0}'.format(source_length)):
            targets_without_start_token = tf.identity(
                targets[:, 1:],
                name='targets_without_start_token'
            )
            batch_loss = tf.contrib.seq2seq.sequence_loss(
                logits=logits,
                targets=targets_without_start_token,
                weights=tf.ones_like(
                    targets_without_start_token,
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
                tf.cast(tf.equal(
                    predictions,
                    targets_without_start_token
                ), tf.int32),
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

    def make_all_graphs(
        self,
        batch_size,
        X_all,
        y_all,
    ):
        """Make all the training graphs given the training inputs."""
        graphs_and_data = []
        sorted_batches = sorted(
            zip(X_all, y_all),
            key=lambda t: t[0].shape[1]
        )
        for x, y in sorted_batches:
            if x.shape[1] <= 2:
                continue
            graphs_and_data.append({
                'source_length': x.shape[1] - 2,
                'target_length': y.shape[1],
                'X': x,
                'y': y,
                'inputs_and_outputs': self.make_training_graph(
                    batch_size,
                    x.shape[1] - 2,
                    y.shape[1],
                )
            })
        return graphs_and_data

    def make_eval_graph(
        self,
        batch_size,
        source_length,
        target_length,
        bos_token_id,
    ):
        """Make all the placeholders and outputs."""
        with tf.name_scope('eval_placeholders_len{0}'.format(source_length)):
            inputs = tf.placeholder(
                dtype=tf.int32,
                shape=[batch_size, source_length],
                name='inputs',
            )
            bos_tokens = tf.constant(
                [bos_token_id] * batch_size,
                dtype=tf.int32,
                shape=[batch_size],
                name='bos_tokens'
            )

        with tf.name_scope('eval_encoder_len{0}'.format(source_length)):
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
            # attended_states will have shape
            # (batch_size, num_steps, hidden_size)
            attended_states = tf.identity(
                tf.einsum(
                    'ij,fgj->fgi',
                    self.attention_params['W'],
                    reshaped_states_encoder,
                ),
                name='attended_states',
            )
            # final_states will have shape
            # (batch_size, hidden_size)
            final_states = h_states_encoder[-1]

        with tf.name_scope('eval_decoder_len{0}'.format(source_length)):
            # embedded_decoder_inputs = tf.nn.embedding_lookup(
            #     self.target_embedding_matrix,
            #     targets,
            #     name='embedded_decoder_inputs',
            # )
            transposed_target_embeddings = tf.transpose(
                self.target_embedding_matrix,
                [1, 0],
                'transposed_target_embeddings',
            )
            h_prev_decoder = final_states
            prev_outputs = bos_tokens
            attention_weights = []
            output_tokens = []
            for i in range(target_length):
                embedded_decoder_inputs = tf.nn.embedding_lookup(
                    self.target_embedding_matrix,
                    prev_outputs,
                    name='embedded_decoder_inputs{0}'.format(i),
                )
                # attention_weights_unnormalized will have shape
                # (batch_size, source_length)
                attention_weights_unnormalized = tf.identity(
                    tf.einsum(
                        'ik,ijk->ij',
                        h_prev_decoder,
                        attended_states,
                    ),
                    name='attention_weights_unnormalized{0}'.format(i)
                )
                # attention_weights_normalized will have shape
                # (batch_size, source_length)
                attention_weights_normalized = tf.nn.softmax(
                    attention_weights_unnormalized,
                    name='attention_weights_normalized{0}'.format(i)
                )
                attention_weights.append(attention_weights_normalized)
                # context_vector will have shape
                # (batch_size, hidden_size)
                context_vector = tf.identity(
                    tf.einsum(
                        'ij,ijk->ik',
                        attention_weights_normalized,
                        reshaped_states_encoder,
                    ),
                    name='context_vector{0}'.format(i)
                )
                h_states = self._attn_gru_update(
                    embedded_decoder_inputs,
                    context_vector,
                    h_prev_decoder,
                    i
                )
                antiembeddings = tf.nn.xw_plus_b(
                    h_states,
                    self.softmax_params['W'],
                    self.softmax_params['b'],
                    name='antiembeddings{0}'.format(i),
                )
                logits = tf.matmul(
                    antiembeddings,
                    transposed_target_embeddings,
                    name='logits{0}'.format(i),
                )
                output_tokens.append(
                    tf.argmax(
                        logits,
                        axis=1,
                        name='output{0}'.format(i),
                    )
                )

                h_prev_decoder = h_states
                prev_outputs = output_tokens[-1]
            # concatenated_attention_weights will have shape
            # (batch_size, (target_length - 1) * source_length)
            concatenated_attention_weights = tf.concat(
                attention_weights,
                axis=1,
                name='concatenated_attention_weights'
            )
            # reshaped_attention_weights will have have shape
            # (batch_size, (target_length - 1) * source_length)
            reshaped_attention_weights = tf.reshape(
                concatenated_attention_weights,
                [batch_size, target_length, source_length],
                name='attention_weights',
            )
            outputs = tf.stack(
                output_tokens,
                axis=1,
                name='eval_outputs'
            )

        return {
            'placeholders': {
                'inputs': inputs,
            },
            'outputs': {
                'outputs': outputs,
                'attention_weights': reshaped_attention_weights,
            },
        }


def generate_training_epoch(X, y, batch_size):
    pass
