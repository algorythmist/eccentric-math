import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt


def fractional_derivative(p: Polynomial, a: float) -> Polynomial:
    """
    Calculate the fractional derivative of a polynomial
    :param p: A polynomial
    :param a: The degree of the fractional derivative
    :return:
    """
    coeff = 1
    result = 0
    # Evaluate the integer part of the derivative
    n = int(np.floor(a))
    f = p.deriv(n) if n > 0 else p
    # the remainder is a numer between 0 and 1
    a = a - n
    for n in range(0, f.degree()):
        f = f.deriv(n)
        # Update the term for the current n
        coeff *= (a - n) / (n+1)
        # Add the term to the result
        result += coeff * f
    return result


def plot_polynomial(p, points=100):
    x = np.linspace(p.domain[0], p.domain[1], points)
    return plt.plot(x, p(x))


if __name__ == '__main__':

    p = Polynomial.fromroots([-1.5,0,1], domain =[-2,2])
    print(p.degree())
    f = fractional_derivative(p, 0.5)
    plot_polynomial(f)
    plt.show()