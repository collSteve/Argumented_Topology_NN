import numpy as np


loss_funcs = {
    "mse": {
        "func": (lambda y, y_hat: np.sum((y - y_hat) ** 2)),
        "diff": (lambda y, y_hat: -2 * (y - y_hat))
        }
}