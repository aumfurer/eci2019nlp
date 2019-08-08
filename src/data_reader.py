import os

__base = os.path.dirname(os.path.dirname(__file__))


def data(*xs):
    return os.path.join(__base, 'data', *xs)
