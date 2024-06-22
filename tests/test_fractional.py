from fractional import *


def test_linear_polynomial():
    """
    Verify that D^a x = ax
    :return:
    """
    p = Polynomial([0, 1])
    assert Polynomial([0, 0.5]) == fractional_derivative(p, 0.5)
    assert Polynomial([0, 0.2]) == fractional_derivative(p, 0.2)


def test_quadratic_polynomial():
    p = Polynomial([0, 0, 0.5])
    assert Polynomial([0, 0.5]) == fractional_derivative(p, 1.5)
    p = Polynomial([1, 1, 1])
    assert Polynomial([0.375, 0.25, 0.5]) == fractional_derivative(p, 0.5)
