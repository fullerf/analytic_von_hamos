import tensorflow as tf
import numpy as np

#
__all__ = ['dety', 'detz']
#
pi = float(np.pi)

def dety(Cap_Psi,a,b,c,p,q,r,t,u,v,r0,Theta):
    sqrt_two = tf.sqrt(tf.constant(2.,dtype=tf.float64))
    return -((tf.math.sqrt((t**2 + v**2) * (t**2 + u**2 +   v**2)) * (2 * b * r0 * t * tf.math.cos(Theta) + 2 * q * r0 * t * tf.math.cos(Theta) +   2 * b * p * t * tf.math.cos(2 * Theta) + 2 * a * q * t * tf.math.cos(2 * Theta) -   2 * b * r0 * t * tf.math.cos(2 * Theta) - 2 * q * r0 * t * tf.math.cos(2 * Theta) -   2 * b * c * v * tf.math.cos(2 * Theta) + 2 * b * r * v * tf.math.cos(2 * Theta) -   2 * a * r0 * t * tf.math.sin(Theta) - 2 * p * r0 * t * tf.math.sin(Theta) +   4 * r0**2 * t * tf.math.sin(Theta) + 2 * c * r0 * v * tf.math.sin(Theta) -   2 * r * r0 * v * tf.math.sin(Theta) - 2 * a * p * t * tf.math.sin(2 * Theta) +   2 * b * q * t * tf.math.sin(2 * Theta) + 2 * a * r0 * t * tf.math.sin(2 * Theta) +   2 * p * r0 * t * tf.math.sin(2 * Theta) - 2 * r0**2 * t * tf.math.sin(2 * Theta) +   2 * a * c * v * tf.math.sin(2 * Theta) - 2 * a * r * v * tf.math.sin(2 * Theta) -   2 * c * r0 * v * tf.math.sin(2 * Theta) + 2 * r * r0 * v * tf.math.sin(2 * Theta) +   sqrt_two * q * v * tf.math.sqrt((tf.math.sin(Cap_Psi)**(-1))**2 * (r0**2 + (a**2 - b**2 -   2 * a * r0 + r0**2) * tf.math.cos(2 * Theta) +   4 * (a - r0) * r0 * tf.math.cos(Theta) * tf.math.cos(Cap_Psi)**2 +   a**2 * tf.math.cos(2 * Cap_Psi) + b**2 * tf.math.cos(2 * Cap_Psi) -   2 * a * r0 * tf.math.cos(2 * Cap_Psi) +   2 * r0**2 * tf.math.cos(2 * Cap_Psi) + 2 * b * r0 * tf.math.sin(Theta) +   2 * b * r0 * tf.math.cos(2 * Cap_Psi) * tf.math.sin(Theta) +   2 * a * b * tf.math.sin(2 * Theta) - 2 * b * r0 * tf.math.sin(2 * Theta))) -   sqrt_two * b * v * tf.math.cos(2 * Theta) * tf.math.sqrt((tf.math.sin(Cap_Psi)**(-1))**2 * (r0**2 + (a**2 -   b**2 - 2 * a * r0 + r0**2) * tf.math.cos(2 * Theta) +   4 * (a - r0) * r0 * tf.math.cos(Theta) * tf.math.cos(Cap_Psi)**2 +   a**2 * tf.math.cos(2 * Cap_Psi) + b**2 * tf.math.cos(2 * Cap_Psi) -   2 * a * r0 * tf.math.cos(2 * Cap_Psi) +   2 * r0**2 * tf.math.cos(2 * Cap_Psi) + 2 * b * r0 * tf.math.sin(Theta) +   2 * b * r0 * tf.math.cos(2 * Cap_Psi) * tf.math.sin(Theta) +   2 * a * b * tf.math.sin(2 * Theta) - 2 * b * r0 * tf.math.sin(2 * Theta))) +   2 * sqrt_two * r0 * v * tf.math.sin(Theta) * tf.math.sqrt((tf.math.sin(Cap_Psi)**(-1))**2 * (r0**2 + (a**2 - b**2 - 2 * a * r0 + r0**2) * tf.math.cos(2 * Theta) +   4 * (a - r0) * r0 * tf.math.cos(Theta) * tf.math.cos(Cap_Psi)**2 +   a**2 * tf.math.cos(2 * Cap_Psi) + b**2 * tf.math.cos(2 * Cap_Psi) -   2 * a * r0 * tf.math.cos(2 * Cap_Psi) +   2 * r0**2 * tf.math.cos(2 * Cap_Psi) + 2 * b * r0 * tf.math.sin(Theta) +   2 * b * r0 * tf.math.cos(2 * Cap_Psi) * tf.math.sin(Theta) +   2 * a * b * tf.math.sin(2 * Theta) - 2 * b * r0 * tf.math.sin(2 * Theta))) +   sqrt_two * a * v * tf.math.sin(2 * Theta) * tf.math.sqrt((tf.math.sin(Cap_Psi)**(-1))**2 * (r0**2 + (a**2 -   b**2 - 2 * a * r0 + r0**2) * tf.math.cos(2 * Theta) +   4 * (a - r0) * r0 * tf.math.cos(Theta) * tf.math.cos(Cap_Psi)**2 +   a**2 * tf.math.cos(2 * Cap_Psi) + b**2 * tf.math.cos(2 * Cap_Psi) -   2 * a * r0 * tf.math.cos(2 * Cap_Psi) +   2 * r0**2 * tf.math.cos(2 * Cap_Psi) + 2 * b * r0 * tf.math.sin(Theta) +   2 * b * r0 * tf.math.cos(2 * Cap_Psi) * tf.math.sin(Theta) +   2 * a * b * tf.math.sin(2 * Theta) - 2 * b * r0 * tf.math.sin(2 * Theta))) -   sqrt_two * r0 * v * tf.math.sin(2 * Theta) * tf.math.sqrt((tf.math.sin(Cap_Psi)**(-1))**2 * (r0**2 + (a**2 -   b**2 - 2 * a * r0 + r0**2) * tf.math.cos(2 * Theta) +   4 * (a - r0) * r0 * tf.math.cos(Theta) * tf.math.cos(Cap_Psi)**2 +   a**2 * tf.math.cos(2 * Cap_Psi) + b**2 * tf.math.cos(2 * Cap_Psi) -   2 * a * r0 * tf.math.cos(2 * Cap_Psi) +   2 * r0**2 * tf.math.cos(2 * Cap_Psi) + 2 * b * r0 * tf.math.sin(Theta) +   2 * b * r0 * tf.math.cos(2 * Cap_Psi) * tf.math.sin(Theta) +   2 * a * b * tf.math.sin(2 * Theta) -   2 * b * r0 * tf.math.sin(2 * Theta)))))/((t**2 +   v**2) * (2 * a * t * tf.math.cos(2 * Theta) - 2 * r0 * t * tf.math.cos(2 * Theta) -   2 * b * u * tf.math.cos(2 * Theta) + 2 * r0 * u * tf.math.sin(Theta) +   2 * tf.math.cos(Theta) * (r0 * t + 2 * (b * t + (a - r0) * u) * tf.math.sin(Theta)) +   sqrt_two * v * tf.math.sqrt((tf.math.sin(Cap_Psi)**(-1))**2 * (r0**2 + (a**2 - b**2 -   2 * a * r0 + r0**2) * tf.math.cos(2 * Theta) +   4 * (a - r0) * r0 * tf.math.cos(Theta) * tf.math.cos(Cap_Psi)**2 +   a**2 * tf.math.cos(2 * Cap_Psi) + b**2 * tf.math.cos(2 * Cap_Psi) -   2 * a * r0 * tf.math.cos(2 * Cap_Psi) +   2 * r0**2 * tf.math.cos(2 * Cap_Psi) + 2 * b * r0 * tf.math.sin(Theta) +   2 * b * r0 * tf.math.cos(2 * Cap_Psi) * tf.math.sin(Theta) +   2 * a * b * tf.math.sin(2 * Theta) - 2 * b * r0 * tf.math.sin(2 * Theta)))))) 
def detz(Cap_Psi,a,b,c,p,q,r,t,u,v,r0,Theta):
    sqrt_two = tf.sqrt(tf.constant(2.,dtype=tf.float64))
    return -(1/tf.math.sqrt(t**2 + v**2))  * v * (-p + r0 -   r0 * tf.math.cos(Theta) + ((r0 * tf.math.cos(Theta) + (a - r0) * tf.math.cos(2 * Theta) + b * tf.math.sin(2 * Theta)) * (2 * p * t - 2 * r0 * t +   2 * q * u - 2 * c * v + 2 * r * v + 2 * r0 * t * tf.math.cos(Theta) +   2 * r0 * u * tf.math.sin(Theta) -   sqrt_two * v * tf.math.sqrt((tf.math.sin(Cap_Psi)**(-1))**2 * (r0**2 + (a**2 - b**2 -   2 * a * r0 + r0**2) * tf.math.cos(2 * Theta) +   4 * (a - r0) * r0 * tf.math.cos(Theta) * tf.math.cos(Cap_Psi)**2 +   a**2 * tf.math.cos(2 * Cap_Psi) + b**2 * tf.math.cos(2 * Cap_Psi) -   2 * a * r0 * tf.math.cos(2 * Cap_Psi) +   2 * r0**2 * tf.math.cos(2 * Cap_Psi) + 2 * b * r0 * tf.math.sin(Theta) +   2 * b * r0 * tf.math.cos(2 * Cap_Psi) * tf.math.sin(Theta) +   2 * a * b * tf.math.sin(2 * Theta) -   2 * b * r0 * tf.math.sin(2 * Theta)))))/(2 * a * t * tf.math.cos(2 * Theta) -   2 * r0 * t * tf.math.cos(2 * Theta) - 2 * b * u * tf.math.cos(2 * Theta) +   2 * r0 * u * tf.math.sin(Theta) +   2 * tf.math.cos(Theta) * (r0 * t + 2 * (b * t + (a - r0) * u) * tf.math.sin(Theta)) +   sqrt_two * v * tf.math.sqrt((tf.math.sin(Cap_Psi)**(-1))**2 * (r0**2 + (a**2 - b**2 -   2 * a * r0 + r0**2) * tf.math.cos(2 * Theta) +   4 * (a - r0) * r0 * tf.math.cos(Theta) * tf.math.cos(Cap_Psi)**2 +   a**2 * tf.math.cos(2 * Cap_Psi) + b**2 * tf.math.cos(2 * Cap_Psi) -   2 * a * r0 * tf.math.cos(2 * Cap_Psi) +   2 * r0**2 * tf.math.cos(2 * Cap_Psi) + 2 * b * r0 * tf.math.sin(Theta) +   2 * b * r0 * tf.math.cos(2 * Cap_Psi) * tf.math.sin(Theta) +   2 * a * b * tf.math.sin(2 * Theta) - 2 * b * r0 * tf.math.sin(2 * Theta))))) +   1/tf.math.sqrt(t**2 + v**2) * t * (c - r +   1/sqrt_two * (tf.math.sqrt((tf.math.sin(Cap_Psi)**(-1))**2 * (r0**2 + (a**2 - b**2 -   2 * a * r0 + r0**2) * tf.math.cos(2 * Theta) +   4 * (a - r0) * r0 * tf.math.cos(Theta) * tf.math.cos(Cap_Psi)**2 +   a**2 * tf.math.cos(2 * Cap_Psi) + b**2 * tf.math.cos(2 * Cap_Psi) -   2 * a * r0 * tf.math.cos(2 * Cap_Psi) +   2 * r0**2 * tf.math.cos(2 * Cap_Psi) + 2 * b * r0 * tf.math.sin(Theta) +   2 * b * r0 * tf.math.cos(2 * Cap_Psi) * tf.math.sin(Theta) +   2 * a * b * tf.math.sin(2 * Theta) -   2 * b * r0 * tf.math.sin(2 * Theta)))) - (tf.math.sqrt((tf.math.sin(Cap_Psi)**(-1))**2 * (r0**2 + (a**2 - b**2 - 2 * a * r0 + r0**2) * tf.math.cos(2 * Theta) +   4 * (a - r0) * r0 * tf.math.cos(Theta) * tf.math.cos(Cap_Psi)**2 +   a**2 * tf.math.cos(2 * Cap_Psi) + b**2 * tf.math.cos(2 * Cap_Psi) -   2 * a * r0 * tf.math.cos(2 * Cap_Psi) +   2 * r0**2 * tf.math.cos(2 * Cap_Psi) + 2 * b * r0 * tf.math.sin(Theta) +   2 * b * r0 * tf.math.cos(2 * Cap_Psi) * tf.math.sin(Theta) +   2 * a * b * tf.math.sin(2 * Theta) -   2 * b * r0 * tf.math.sin(2 * Theta))) * (-2 * p * t + 2 * r0 * t - 2 * q * u +   2 * c * v - 2 * r * v - 2 * r0 * t * tf.math.cos(Theta) -   2 * r0 * u * tf.math.sin(Theta) +   sqrt_two * v * tf.math.sqrt((tf.math.sin(Cap_Psi)**(-1))**2 * (r0**2 + (a**2 - b**2 -   2 * a * r0 + r0**2) * tf.math.cos(2 * Theta) +   4 * (a - r0) * r0 * tf.math.cos(Theta) * tf.math.cos(Cap_Psi)**2 +   a**2 * tf.math.cos(2 * Cap_Psi) + b**2 * tf.math.cos(2 * Cap_Psi) -   2 * a * r0 * tf.math.cos(2 * Cap_Psi) +   2 * r0**2 * tf.math.cos(2 * Cap_Psi) + 2 * b * r0 * tf.math.sin(Theta) +   2 * b * r0 * tf.math.cos(2 * Cap_Psi) * tf.math.sin(Theta) +   2 * a * b * tf.math.sin(2 * Theta) -   2 * b * r0 * tf.math.sin(2 * Theta)))))/(sqrt_two * (2 * a * t * tf.math.cos(2 * Theta) - 2 * r0 * t * tf.math.cos(2 * Theta) -   2 * b * u * tf.math.cos(2 * Theta) + 2 * r0 * u * tf.math.sin(Theta) +   2 * tf.math.cos(Theta) * (r0 * t +   2 * (b * t + (a - r0) * u) * tf.math.sin(Theta)) +   sqrt_two * v * tf.math.sqrt((tf.math.sin(Cap_Psi)**(-1))**2 * (r0**2 + (a**2 - b**2 -   2 * a * r0 + r0**2) * tf.math.cos(2 * Theta) +   4 * (a - r0) * r0 * tf.math.cos(Theta) * tf.math.cos(Cap_Psi)**2 +   a**2 * tf.math.cos(2 * Cap_Psi) + b**2 * tf.math.cos(2 * Cap_Psi) -   2 * a * r0 * tf.math.cos(2 * Cap_Psi) +   2 * r0**2 * tf.math.cos(2 * Cap_Psi) + 2 * b * r0 * tf.math.sin(Theta) +   2 * b * r0 * tf.math.cos(2 * Cap_Psi) * tf.math.sin(Theta) +   2 * a * b * tf.math.sin(2 * Theta) - 2 * b * r0 * tf.math.sin(2 * Theta))))))
