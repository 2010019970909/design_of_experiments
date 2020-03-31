"""
The aim of this script is to automate some process in
the Design of experiments (DoE) workflow.
"""
__author__ = "Vincent STRAGIER"

# Maths modules
from itertools import permutations, combinations
from scipy.special import erfinv
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
    Generate the X matrix to compute the a_i coefficents for a 2**n DoE.

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


def draw_coefficents(mpl, coefficents, coefficents_labels=None, remove_a0: bool = False, title: str = "Coefficients bar chart", legend: str = "Coefficients", draw: bool = True, **kwargs):
    """
    Draw the bar chart of the coefficients a_i.

    coefficents:
    A list or an array with the coefficients.

    coefficents_labels:
    A list or an array with the labels of the coefficient.

    title:
    The title of the chart.

    legend:
    Legend to display on the chart.

    draw:
    Defines if the figure has to be displayed or no.

    **kwargs:
    Others optional arguments for the plot function (like the color, etc)
    """
    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    x = np.arange(len(coefficents))
    n = int(np.log2(len(coefficents)))

    if coefficents_labels:
        labels = coefficents_labels
    else:
        labels = gen_a_labels(n)

    if remove_a0:
        coefficents = coefficents[1:]
        labels = labels[1:]
        x = np.arange(len(coefficents))

    # mpl.figure()
    mpl.ax.clear()
    rects = mpl.ax.bar(x, coefficents, **kwargs)

    for rect in rects:
        height = rect.get_height()
        if height < 0:
            va = 'top'
            xytext = (0, -3)
        else:
            va = 'bottom'
            xytext = (0, 3)

        mpl.ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=xytext,  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va=va)

    mpl.ax.set_title(title)
    mpl.ax.set_xticks(x)
    mpl.ax.set_xticklabels(labels)
    # mpl.ax.grid(which='major')
    mpl.ax.legend([legend])
    # mpl.tight_layout()
    if draw:
        mpl.draw()


def plot_coefficents(coefficents, coefficents_labels=None, remove_a0: bool = False, title: str = "Coefficients bar chart", legend: str = "Coefficients", block: bool = False, show: bool = False, **kwargs):
    """
    Plot the bar chart of the coefficients a_i.

    coefficents:
    A list or an array with the coefficients.

    coefficents_labels:
    A list or an array with the labels of the coefficient.

    title:
    The title of the chart.

    legend:
    Legend to display on the chart.

    block:
    Defines if the plot should block or no the execution of the code.

    show:
    Defines if the figure has to be displayed or no.

    **kwargs:
    Others optional arguments for the plot function (like the color, etc)
    """
    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    x = np.arange(len(coefficents))
    n = int(np.log2(len(coefficents)))

    if coefficents_labels:
        labels = coefficents_labels
    else:
        labels = gen_a_labels(n)

    if remove_a0:
        coefficents = coefficents[1:]
        labels = labels[1:]
        x = np.arange(len(coefficents))

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
    if show:
        plt.show(block=block)

    return fig, ax


def draw_pareto(mpl, coefficents, coefficents_labels=None, remove_a0: bool = True, title: str = "Pareto bar chart", legend: str = "| Coefficients |", draw: bool = True, **kwargs):
    """
    Draw the Pareto's bar chart of the coefficients a_i.

    coefficents:
    A list or an array with the coefficients.

    coefficents_labels:
    A list or an array with the labels of the coefficient.

    title:
    The title of the chart.

    legend:
    Legend to display on the chart.

    draw:
    Defines if the figure has to be displayed or no.

    **kwargs:
    Others optional argumentd for the plot function (like the color, etc).
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

    if remove_a0:
        coefficents = coefficents[1:]
        labels = labels[1:]
        y = np.arange(len(coefficents))

    # https://stackoverflow.com/a/7851166
    index = sorted(range(len(coefficents)),
                   key=coefficents.__getitem__, reverse=True)
    coefficents = coefficents[index]
    labels = labels[index]

    # mpl.figure()
    mpl.ax.clear()
    rects = mpl.ax.barh(y, coefficents, **kwargs)

    i = 0

    for rect in rects:
        x = rect.get_width()

        va = 'center'

        if i == 0:
            xytext = (-4*len(str(x)), 0)
        else:
            xytext = (4*len(str(x)), 0)

        mpl.ax.annotate('{}'.format(x),
                        xy=(x, i),
                        xytext=xytext,  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va=va)
        i += 1

    mpl.ax.set_title(title)
    mpl.ax.set_yticks(y)
    mpl.ax.set_yticklabels(labels)
    # ax.grid(which='major')
    mpl.ax.legend([legend])
    # mpl.ax.autoscale_view(True,True,True)
    # fig.tight_layout()
    if draw:
        mpl.draw()


def plot_pareto(coefficents, coefficents_labels=None, remove_a0: bool = True, title: str = "Pareto bar chart", legend: str = "| Coefficients |", block: bool = False, show: bool = False, **kwargs):
    """
    Plot the Pareto's bar chart of the coefficients a_i.

    coefficents:
    A list or an array with the coefficients.

    coefficents_labels:
    A list or an array with the labels of the coefficient.

    title:
    The title of the chart.

    legend:
    Legend to display on the chart.

    block:
    Defines if the plot should block or no the execution of the code.

    show:
    Defines if the figure has to be displayed or no.

    **kwargs:
    Others optional argumentd for the plot function (like the color, etc).
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

    if remove_a0:
        coefficents = coefficents[1:]
        labels = labels[1:]
        y = np.arange(len(coefficents))

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
    if show:
        plt.show(block=block)

    return fig, ax


def draw_henry(mpl, coefficents, coefficents_labels=None, remove_a0: bool = True, empirical_cumulative_distribution: str = "classical", a: float = 0, title: str = "Henry bar chart", legend: str = "| Coefficients |", draw: bool = True, **kwargs):
    """
    Draw the Henry's chart of the coefficients a_i.

    coefficents:
    A list or an array with the coefficients.

    coefficents_labels:
    A list or an array with the labels of the coefficient.

    empirical_cumulative_distribution:

    classical - f(i) = i/N

    modified - f(i) = (i + a)/(N + 1 + 2a)

    title:
    The title of the chart.

    legend:
    Legend to display on the chart.

    draw:
    Defines if the figure has to be displayed or no.

    **kwargs:
    Others optional arguments for the plot function (like the color, etc).
    """
    l = len(coefficents)
    n = int(np.log2(l))

    if coefficents_labels:
        labels = np.array(coefficents_labels, dtype=str)
    else:
        labels = np.array(gen_a_labels(n), dtype=str)

    if remove_a0:
        coefficents = coefficents[1:]
        labels = labels[1:]
        l = len(coefficents)

    # https://stackoverflow.com/a/7851166
    index = sorted(range(len(coefficents)),
                   key=coefficents.__getitem__, reverse=False)
    coefficents = coefficents[index]
    labels = labels[index]

    # Empirical cumulative distribution f(i)
    dist = coefficents

    if empirical_cumulative_distribution == "classical":
        for i in range(l):
            dist[i] = (i+1)/l
    elif empirical_cumulative_distribution == "modified":
        for i in range(l):
            dist[i] = (i+1+a)/(l+1+2*a)
    else:
        print("Error: unknown empirical mode.")

    # Corresponding quantile (normit) z(i)
    normits = erfinv(2*dist - 1) * np.sqrt(2)

    # mpl.figure()
    mpl.ax.clear()
    mpl.ax.plot(coefficents, normits, marker='1',
                linestyle='--', linewidth=0.5, **kwargs)

    mpl.ax.set_title(title)
    mpl.ax.set_yticks(normits)
    mpl.ax.set_yticklabels(labels)
    mpl.ax.grid(which='major')
    mpl.ax.legend([legend])
    # fig.tight_layout()
    if draw:
        mpl.draw()


def plot_henry(coefficents, coefficents_labels=None, remove_a0: bool = True, empirical_cumulative_distribution: str = "classical", a: float = 0, title: str = "Henry bar chart", legend: str = "| Coefficients |", block: bool = False, show: bool = False, **kwargs):
    """
    Plot the Henry's chart of the coefficients a_i.

    coefficents:
    A list or an array with the coefficients.

    coefficents_labels:
    A list or an array with the labels of the coefficient.

    empirical_cumulative_distribution:

    classical - f(i) = i/N

    modified - f(i) = (i + a)/(N + 1 + 2a)

    title:
    The title of the chart.

    legend:
    Legend to display on the chart.

    block:
    Defines if the plot should block or no the execution of the code.

    show:
    Defines if the figure has to be displayed or no.

    **kwargs:
    Others optional arguments for the plot function (like the color, etc).
    """
    l = len(coefficents)
    n = int(np.log2(l))

    if coefficents_labels:
        labels = np.array(coefficents_labels, dtype=str)
    else:
        labels = np.array(gen_a_labels(n), dtype=str)

    if remove_a0:
        coefficents = coefficents[1:]
        labels = labels[1:]
        l = len(coefficents)

    # https://stackoverflow.com/a/7851166
    index = sorted(range(len(coefficents)),
                   key=coefficents.__getitem__, reverse=False)
    coefficents = coefficents[index]
    labels = labels[index]

    # Empirical cumulative distribution f(i)
    dist = coefficents

    if empirical_cumulative_distribution == "classical":
        for i in range(l):
            dist[i] = (i+1)/l
    elif empirical_cumulative_distribution == "modified":
        for i in range(l):
            dist[i] = (i+1+a)/(l+1+2*a)
    else:
        print("Error: unknown empirical mode.")

    # Corresponding quantile (normit) z(i)
    normits = erfinv(2*dist - 1) * np.sqrt(2)

    fig, ax = plt.subplots()
    ax.plot(coefficents, normits, marker='1',
            linestyle='--', linewidth=0.5, **kwargs)

    ax.set_title(title)
    ax.set_yticks(normits)
    ax.set_yticklabels(labels)
    ax.grid(which='major')
    ax.legend([legend])
    fig.tight_layout()
    if show:
        plt.show(block=block)

    return fig, ax

def clear_draw(mpl):
    mpl.ax.clear()
    mpl.draw()


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
    plot_henry(a_hat, empirical_cumulative_distribution="modified",
               block=True, color="blue")

    print('Test 1:', y_hat == y, end="\n\n")

    # Test 2
    a_hat = np.array([10.25, 1.25, 0.75, 0.05])
    print("a_hat:", a_hat)

    y = np.array([8.3, 10.7, 9.7, 12.3])
    print("y:", y)

    a_hat_check = np.dot(gen_X_hat(n=2), y)
    print("a_hat_check", a_hat_check)

    plot_coefficents(a_hat, block=True, color="orange")
    plot_henry(a_hat, empirical_cumulative_distribution="modified",
               block=True, color="blue")

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
