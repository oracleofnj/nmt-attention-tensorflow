"""Functional implementation of GRU RNN."""
import tensorflow as tf


def make_rnn_variables(
    vocab_size,
    embedding_size,
    hidden_size,
):
    """Create the trainable variables for the RNN."""
    with tf.variable_scope(
        'RNNParams',
        reuse=False,
        initializer=tf.random_uniform_initializer(-0.05, 0.05),
    ):
        embedding_matrix = tf.get_variable(
            'embedding',
            [vocab_size, embedding_size],
            dtype=tf.float16,
        )
        gru_params = {
            'U_z': tf.get_variable(
                'U_z',
                [hidden_size, hidden_size],
                dtype=tf.float16,
            ),
            'W_z': tf.get_variable(
                'W_z',
                [embedding_size, hidden_size],
                dtype=tf.float16,
            ),
            'b_z': tf.get_variable(
                'b_z',
                [hidden_size],
                dtype=tf.float16,
            ),
            'U_r': tf.get_variable(
                'U_r',
                [hidden_size, hidden_size],
                dtype=tf.float16,
            ),
            'W_r': tf.get_variable(
                'W_r',
                [embedding_size, hidden_size],
                dtype=tf.float16,
            ),
            'b_r': tf.get_variable(
                'b_r',
                [hidden_size],
                dtype=tf.float16,
            ),
            'U_h': tf.get_variable(
                'U_h',
                [hidden_size, hidden_size],
                dtype=tf.float16,
            ),
            'W_h': tf.get_variable(
                'W_h',
                [embedding_size, hidden_size],
                dtype=tf.float16,
            ),
            'b_h': tf.get_variable(
                'b_h',
                [hidden_size],
                dtype=tf.float16,
            ),
        }
        softmax_params = {
            'W': tf.get_variable(
                'softmax_w',
                [hidden_size, vocab_size],
                dtype=tf.float16,
            ),
            'b': tf.get_variable(
                'softmax_b',
                [vocab_size],
                dtype=tf.float16,
            )
        }

    return {
        'embedding_matrix': embedding_matrix,
        'gru_params': gru_params,
        'softmax_params': softmax_params
    }


def gru_update(x_t, h_t_minus_1, gru_params, timestep=None):
    """Return a tensor for the next value of the hidden state."""
    U_z, W_z, b_z = gru_params['U_z'], gru_params['W_z'], gru_params['b_z']
    U_r, W_r, b_r = gru_params['U_r'], gru_params['W_r'], gru_params['b_r']
    U_h, W_h, b_h = gru_params['U_h'], gru_params['W_h'], gru_params['b_h']
    with tf.name_scope('gru_calculations'):
        r_t = tf.sigmoid(
            tf.matmul(x_t, W_r) + tf.matmul(h_t_minus_1, U_r) + b_r,
            name='r' + (
                '_{0}'.format(timestep) if timestep is not None else ''
            ),
        )
        z_t = tf.sigmoid(
            tf.matmul(x_t, W_z) + tf.matmul(h_t_minus_1, U_z) + b_z,
            name='z' + (
                '_{0}'.format(timestep) if timestep is not None else ''
            ),
        )
        h_tilde_t = tf.tanh(
            tf.matmul(x_t, W_h) + tf.matmul(h_t_minus_1 * r_t, U_h) + b_h,
            name='h_tilde' + (
                '_{0}'.format(timestep) if timestep is not None else ''
            ),
        )
        h_t = z_t * h_t_minus_1 + (1 - z_t) * h_tilde_t
    return h_t


def make_rnn_outputs(
    input_sequence,
    vocab_size,
    hidden_size,
    batch_size,
    num_steps,
    rnn_variables,
):
    """Construct the RNN graph."""
    embedding_matrix = rnn_variables['embedding_matrix']
    gru_params = rnn_variables['gru_params']
    softmax_params = rnn_variables['softmax_params']
    with tf.name_scope('RNN'):
        embedded_inputs = tf.nn.embedding_lookup(
            embedding_matrix,
            input_sequence,
            name='embedded_inputs',
        )
        h_start = tf.zeros(
            [batch_size, hidden_size],
            name='h_start',
            dtype=tf.float16,
        )
        h_prev = h_start
        h_states = []
        for i in range(num_steps):
            h_states.append(gru_update(
                embedded_inputs[:, i, :],
                h_prev,
                gru_params,
                i
            ))
            h_prev = h_states[-1]

        # h_states is a list of tensors, each of which has shape
        # (batch_size, hidden_size)
        #
        # we ultimately want to end up with something of shape
        # (batch_size, num_steps, vocab_size)
        #
        # To see why the steps below work, try the following.
        # (In this example, batch_size is 3, hidden_size = 4, num_steps = 2)
        #
        # m1 = tf.constant(np.reshape(np.arange(12),(3,4)))
        # m2 = tf.constant(6 + np.reshape(np.arange(12),(3,4)))
        # concatenated_ms = tf.concat([m1, m2], axis=1)
        # skinny_ms = tf.reshape(concatenated_ms, [-1, 4])
        # reshaped_ms = tf.reshape(skinny_ms, [-1, 2, 4])
        # with tf.Session() as sess:
        #     for m in sess.run([
        #         concatenated_ms,
        #         skinny_ms,
        #         reshaped_ms
        #     ]):
        #         print(m)
        #         print()
        #
        # which prints
        #
        # [[ 0  1  2  3  6  7  8  9]
        #  [ 4  5  6  7 10 11 12 13]
        #  [ 8  9 10 11 14 15 16 17]]
        #
        # [[ 0  1  2  3]
        #  [ 6  7  8  9]
        #  [ 4  5  6  7]
        #  [10 11 12 13]
        #  [ 8  9 10 11]
        #  [14 15 16 17]]
        #
        # [[[ 0  1  2  3]
        #   [ 6  7  8  9]]
        #
        #  [[ 4  5  6  7]
        #   [10 11 12 13]]
        #
        #  [[ 8  9 10 11]
        #   [14 15 16 17]]]
        #
        # concatenated_states will have shape
        # (batch_size, num_steps * hidden_size)
        concatenated_states = tf.concat(
            h_states,
            axis=1,
            name='concatenated_states'
        )
        # reshaped_states (which will get used for attention)
        # will have have shape (batch_size, num_steps, hidden_size)
        reshaped_states = tf.reshape(
            concatenated_states,
            [batch_size, num_steps, hidden_size],
            name='reshaped_states',
        )
        # long_and_skinny_states will have shape
        # (batch_size * num_steps, hidden_size)
        long_and_skinny_states = tf.reshape(
            concatenated_states,
            [batch_size * num_steps, hidden_size],
            name='long_and_skinny_states',
        )
        # long_and_skinny_logits will have shape
        # (batch_size * num_steps, vocab_size)
        long_and_skinny_logits = tf.nn.xw_plus_b(
            long_and_skinny_states,
            softmax_params['W'],
            softmax_params['b'],
            name='long_and_skinny_logits',
        )
        # logits will have shape
        # (batch_size, num_steps, vocab_size)
        logits = tf.reshape(
            long_and_skinny_logits,
            [batch_size, num_steps, vocab_size],
            name='logits'
        )

    return {
        'states': reshaped_states,
        'logits': logits,
    }


def make_placeholders(batch_size, num_steps):
    """Create the placeholder nodes."""
    with tf.name_scope('placeholders'):
        inputs = tf.placeholder(
            dtype=tf.int32,
            shape=[batch_size, num_steps],
            name='inputs',
        )
        targets = tf.placeholder(
            dtype=tf.int32,
            shape=[batch_size, num_steps],
            name='targets',
        )
        learning_rate = tf.placeholder(
            dtype=tf.float16,
            shape=[],
            name='learning_rate',
        )
        max_norm = tf.placeholder(
            dtype=tf.float16,
            shape=[],
            name='max_norm',
        )

    return {
        'inputs': inputs,
        'targets': targets,
        'learning_rate': learning_rate,
        'max_norm': max_norm,
    }


def make_summary_nodes(targets, logits):
    """Create the output nodes."""
    with tf.name_scope('summary'):
        batch_loss = tf.contrib.seq2seq.sequence_loss(
            logits=logits,
            targets=targets,
            weights=tf.ones_like(
                targets,
                dtype=tf.float16
            ),
            average_across_timesteps=False,
            average_across_batch=False,
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

    return {
        'loss': loss,
        'num_correct_predictions': num_correct_predictions,
    }


def make_train_op(loss, learning_rate, max_norm):
    """Create the training op."""
    with tf.name_scope('train'):
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
        'train_op': train_op,
        'gradient_global_norm': gradient_global_norm,
    }
