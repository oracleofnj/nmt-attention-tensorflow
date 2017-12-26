"""Conditional scope entry."""
import tensorflow as tf

# Shoutout to:
# https://stackoverflow.com/questions/41695893/tensorflow-conditionally-add-variable-scope


class empty_scope():
    """Scope that does nothing."""

    def __init__(self):
        """Do nothing."""
        pass

    def __enter__(self):
        """Do nothing."""
        pass

    def __exit__(self, type, value, traceback):
        """Do nothing."""
        pass


def cond_name_scope(scope):
    """Enter a name scope if scope is not None."""
    return empty_scope() if scope is None else tf.name_scope(scope)


def cond_variable_scope(scope):
    """Enter a variable scope if scope is not None."""
    return empty_scope() if scope is None else tf.variable_scope(scope)
