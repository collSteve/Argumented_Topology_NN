import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def diff_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

activation_funcs = {
    "sigmoid": {
        "func": sigmoid,
        "diff": diff_sigmoid
    },
}




