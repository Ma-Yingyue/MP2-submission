"""Submission interface. Implement your own selection method here."""

from threadpoolctl import threadpool_limits

from selector_model import choose, predict_scores


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None, supplied through --checkpoint.
    """
    with threadpool_limits(limits=1, user_api='blas'), threadpool_limits(limits=4, user_api='openmp'):
        return choose(bank, predict_scores(bank, arrays, checkpoint))
