"""Utility functions."""
import tensorflow as tf


def ptb_batcher(raw_data, batch_size, num_steps):
    """Return a batch of data.

    Equivalent of ptb_producer that I wrote to understand all the TF concepts.
    """
    with tf.name_scope('batcher'):
        tf_raw_data = tf.convert_to_tensor(
            raw_data,
            name='raw_data',
            dtype=tf.int32
        )
        data_len = tf.size(
            tf_raw_data,
            name='num_elems'
        )
        num_batches = tf.floordiv(
            data_len, batch_size,
            name='num_batches'
        )
        data = tf.reshape(
            tf_raw_data[:batch_size * num_batches],
            [batch_size, num_batches],
            name='data'
        )
        batches_per_epoch = tf.floordiv(
            num_batches - 1, num_steps,
            name='batches_per_epoch'
        )
        tf_queue = tf.train.range_input_producer(
            limit=batches_per_epoch, shuffle=False
        )
        i = tf_queue.dequeue(name='iter_idx')
        x = tf.identity(
            data[:, (i * num_steps):((i+1) * num_steps)],
            name='x'
        )
        x.set_shape([batch_size, num_steps])
        y = tf.identity(
            data[:, (1 + i * num_steps):(1 + (i+1) * num_steps)],
            name='y'
        )
        y.set_shape([batch_size, num_steps])
        return x, y
