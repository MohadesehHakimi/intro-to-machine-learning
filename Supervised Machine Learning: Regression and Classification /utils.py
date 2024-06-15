import numpy as np


def compute_cost(X, y, w, b):
    """
    J(w, b) = 1/2m * sum((wx + b - y)^2)
    """
    m = len(y)
    J = 1 / (2 * m) * np.sum((w * X + b - y) ** 2)
    return J


def compute_gradient(X, y, w, b):
    """
    dJ(w, b)/dw = 1/m * sum((wx + b - y) * x)
    dJ(w, b)/db = 1/m * sum(wx + b - y)
    """
    m = len(y)
    dw = 1 / m * np.sum((w * X + b - y) * X)
    db = 1 / m * np.sum(w * X + b - y)
    return dw, db


def gradient_descent_univariate(X, y, w, b, alpha, convergence_threshold=1e-6):
    iter_counter = 0
    J = compute_cost(X, y, w, b)
    J_history = [J]
    while True:
        iter_counter += 1
        dw, db = compute_gradient(X, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db
        J_new = compute_cost(X, y, w, b)
        J_history.append(J_new)
        if abs(J - J_new) < convergence_threshold:
            print(f'Converged after {iter_counter} iterations')
            break
        J = J_new
    return w, b, J_history
