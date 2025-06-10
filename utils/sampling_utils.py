import numpy as np


def sample_uniform(list_full):
    # TODO: Change or remove
    return np.random.choice(list_full, 1)[0]


def sample_geometric(list_full, last_frame_probability: float):
    no_frames = len(list_full)
    if no_frames == 1:
        return np.array([1.0])
    probs = np.arange(1, no_frames + 1)
    probs = np.power((1 - last_frame_probability),
                     probs - 1) * last_frame_probability
    probs /= probs.sum()
    return probs
    # return np.random.choice(list_full, p=probs)
