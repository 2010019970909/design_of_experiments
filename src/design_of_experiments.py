"""
The aim of this script is to automate some process in
the Design of experiments (DoE) workflow.
"""
__author__ = "Vincent STRAGIER"

# Maths modules
from itertools import permutations, combinations
# import pandas as pd
import numpy as np

# Plotting module
import matplotlib.pyplot as plt


def gen_design(n: int = 2, perm=None):
    """
    Generate the matrix for factorial design of experiments (2**n)

    n: 
    The number of factors to analyse

    perm: 
    A permutation vector of size 2**n
    """
    set_matrix = set()

    for i in range(n + 1):
        # https://stackoverflow.com/a/41210386 for the permutation
        # https://stackoverflow.com/a/29648719 for the update of the set
        set_matrix.update(set(permutations((n-i)*[-1] + i*[1])))

    # Tranform the matrix to fit the example (Table 10.4.1)
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html to flip the matrix along the Y axis
    if perm:
        return np.flip(np.array(sorted(set_matrix, reverse=True)))[perm]
    return np.flip(np.array(sorted(set_matrix, reverse=True)))


def gen_X(n: int = 2, perm=None, show: bool = False, return_head: bool = False):
    """
    Generate the X matrix to compute the coefficents.

    n: 
    The number of factors to analyse

    perm:
    A permutation vector of size 2**n

    show: 
    If True print the head and the matrix X and return (X, head)
    Else only return X

    return_head: 
    If True, return (X, head)
    """
    DoE = gen_design(n=n, perm=perm)
    X = np.c_[(2**n)*[1], DoE]

    if show:
        head = ['I']
        for i in range(n):
            # Generate the combinations for i position
            combs = sorted(set(combinations(range(1, n+1), i+1)))

            for comb in combs:
                # Generate the column name
                head.append(str(list(comb)).replace('[', '').replace(
                    ']', '').replace(' ', '').replace(',', '•'))

    for i in range(n-1):
        # Generate the combinations for i+2 position
        combs = sorted(set(combinations(range(n), i+2)))

        for comb in combs:
            # Generate the column by combination
            temp_col = (2**n)*[1]
            for j in list(comb):
                temp_col = np.multiply(temp_col, DoE[:, j])
            # Add the column to the matrix
            X = np.c_[X, temp_col]

    if show:
        print(head)
        print(X)
        return X, head

    if return_head:
        return X, head

    return X


def gen_a_labels(n: int = 2):
    """
    Generate a list of labels for the a_i coefficients.

    n: 
    The number of factors to analyse
    """
    head = [r'$\^a_{0}$']
    for i in range(n):
        # Generate the combinations for i position
        combs = sorted(set(combinations(range(1, n+1), i+1)))

        for comb in combs:
            # Generate the column name
            head.append(r"$\^a_{" + str(list(comb)).replace('[', '').replace(
                ']', '').replace(' ', '').replace(',', r' \cdot ') + "}$")

    return head


def gen_X_hat(n: int = 2, perm=None, show: bool = False):
    """
    Generate the matrix X_hat = (X^T * X)^-1 * X^T

    n:
    The number of factors to analyse

    perm:
    A permutation vector of size 2**n

    show:
    If True print the head, the matrix X and X_hat
    Else only return X_hat
    """
    if show:
        X, _ = gen_X(n=n, perm=perm, show=show)
    else:
        X = gen_X(n=n, perm=perm, show=show)

    X_hat = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)

    if show:
        print(X_hat)

    return X_hat


def plot_coefficents(coefficents, coefficents_labels=None, title: str = "Coefficients bar chart", legend: str = "Coefficients", block: bool = False, show: bool = False, **kwargs):
    """
    Plot the bar chart of the coefficients
    """
    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    x = np.arange(len(coefficents))
    n = int(np.log2(len(coefficents)))

    if coefficents_labels:
        labels = coefficents_labels
    else:
        labels = gen_a_labels(n)

    fig, ax = plt.subplots()
    rects = ax.bar(x, coefficents, **kwargs)

    for rect in rects:
        height = rect.get_height()
        if height < 0:
            va = 'top'
            xytext = (0, -3)
        else:
            va = 'bottom'
            xytext = (0, 3)

        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=xytext,  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va=va)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # ax.grid(which='major')
    ax.legend([legend])
    fig.tight_layout()
    plt.show(block=block)

    return fig


def plot_pareto(coefficents, coefficents_labels=None, title: str = "Pareto bar chart", legend: str = "| Coefficients |", block: bool = False, show: bool = False, **kwargs):
    """
    Plot the Pareto bar chart of the coefficients
    """
    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    l = len(coefficents)
    y = np.arange(l)
    n = int(np.log2(l))
    coefficents = np.abs(coefficents)

    if coefficents_labels:
        labels = np.array(coefficents_labels, dtype=str)
    else:
        labels = np.array(gen_a_labels(n), dtype=str)

    # https://stackoverflow.com/a/7851166
    index = sorted(range(len(coefficents)),
                   key=coefficents.__getitem__, reverse=True)
    coefficents = coefficents[index]
    labels = labels[index]

    fig, ax = plt.subplots()
    rects = ax.barh(y, coefficents, **kwargs)

    i = 0

    for rect in rects:
        x = rect.get_width()

        va = 'center'

        if i == 0:
            xytext = (-4*len(str(x)), 0)
        else:
            xytext = (4*len(str(x)), 0)

        ax.annotate('{}'.format(x),
                    xy=(x, i),
                    xytext=xytext,  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va=va)
        i += 1

    ax.set_title(title)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    # ax.grid(which='major')
    ax.legend([legend])
    fig.tight_layout()
    plt.show(block=block)

    return fig


def main():
    # Test 1
    y = np.array([77, 28.5, 141, 110, 161, 113, 220, 190])
    print("y:", y)

    a_hat = np.dot(gen_X_hat(int(np.log2(len(y)))), y)
    print("a_hat:", a_hat)

    y_hat = np.dot(gen_X(n=3), a_hat)
    print("y_hat:", y_hat)

    plot_coefficents(a_hat, block=False, color="orange")
    plot_pareto(a_hat, block=True, color="orange")

    print('Test 1:', y_hat == y, end="\n\n")

    # Test 2
    a_hat = np.array([10.25, 1.25, 0.75, 0.05])
    print("a_hat:", a_hat)

    y = np.array([8.3, 10.7, 9.7, 12.3])
    print("y:", y)

    a_hat_check = np.dot(gen_X_hat(n=2), y)
    print("a_hat_check", a_hat_check)

    #plot_coefficents(a_hat, block=True, color="orange")

    print('Test 2:', a_hat_check == a_hat, end="\n\n")

    # Gen label
    print(gen_a_labels(2)[3])

    """
    n = 3
    
    DoE = gen_design(n)
    print(DoE)

    DoE = gen_design(n, perm=None) # [0, 2, 1, 4, 3, 5, 6, 7])
    print(DoE)

    X = gen_X(n, show=False)
    
    X_hat = gen_X_hat(n, show=True)
    """


if __name__ == "__main__":
    main()
