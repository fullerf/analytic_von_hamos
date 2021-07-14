# import xrt.backends.raycing.materials as mat
import numpy as np
import tensorflow as tf

__all__ = [
    'CrystalGe', 'CrystalSi', 'Ge440Bragg'
]

# +
class CrystalGe(object):
    def __init__(self, h, k, l):
        self.a = 5.65735
        self.d = np.sqrt(self.a**2/(h**2 + k**2 + l**2))
    def __call__(self, E, order=1):
        CH = 12398.419297617678
        return tf.math.asin(tf.clip_by_value(order * CH / (2*self.d*E),-1+1E-16,1-1E-16))
    
class CrystalSi(object):
    def __init__(self, h, k, l):
        self.a = 5.4307717932001225
        self.d = np.sqrt(self.a**2/(h**2 + k**2 + l**2))
    def __call__(self, E, order=1):
        CH = 12398.419297617678
        return tf.math.asin(tf.clip_by_value(order * CH / (2*self.d*E),-1+1E-16,1-1E-16))


# -

def Ge440Bragg(eV):
    return CrystalGe(4,4,0)(eV)
