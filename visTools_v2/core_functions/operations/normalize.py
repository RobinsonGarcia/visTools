import numpy as np

def norm(x):
    return (x - x.min())/(x.max()-x.min())
