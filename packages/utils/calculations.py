from numpy import dot, sqrt
import math

def cos(vA,vB):
    """
    regular cosine similarity for two vectors 'vA' and 'vB'
    """
    denom = (sqrt(dot(vA,vA)) * sqrt(dot(vB,vB)))
    if denom == 0:
        print("Denom is 0, setting to 1")
        denom = 1
    return dot(vA, vB) / denom

def roundup(x):
    return int(math.ceil(x / 9.0)) * 10

def round_down(x):
    return int(math.floor(x / 9.0)) * 10