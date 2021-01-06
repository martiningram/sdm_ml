from itertools import islice
import numpy as np


def split_every(n, iterable):

    # Credit to:
    # https://stackoverflow.com/questions/1915170/split-a-generator-iterable-every-n-items-in-python-splitevery
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


def evaluate_on_chunks(fun, chunk_size, *args, is_df=True):

    indices = np.arange(args[0].shape[0])

    chunk_results = list()

    for cur_indices in split_every(chunk_size, indices):

        cur_indices = list(cur_indices)
        cur_subsets = [x.iloc[cur_indices] if is_df else x[cur_indices] for x in args]
        cur_result = fun(*cur_subsets)
        chunk_results.append(cur_result)

    return np.concatenate(chunk_results)
