from .analytic import *
import tensorflow as tf
import gpflow
import numpy as np
from .bragg import CrystalGe, CrystalSi
import tensorflow_probability as tfp
tfb = tfp.bijectors

__all__ = ["TwoLineFit"]



def batched_closest_and_next_point_on_manifold(manifold_points, test_points):
    M = tf.shape(test_points)[0]
    B = tf.shape(manifold_points)[0] 
    manifold_points_e = tf.tile(manifold_points[None,:,:,:],[M,1,1,1])
    closest_index = tf.argmin(tf.reduce_sum(tf.math.square(manifold_points - 
                                                      test_points[:,None,None,:]),-1),-1)
    N = tf.cast(tf.shape(manifold_points)[1],closest_index.dtype)
    aug_closest_index = tf.stack([
                            tf.tile(tf.range(M,dtype=tf.int64)[:,None],[1,B]),
                            tf.tile(tf.range(B,dtype=tf.int64)[None,:],[M,1]),
                            closest_index
                        ],-1) #MxBx2
    manifold_points_e2 = tf.tile(manifold_points[None,:,:,:],[M,1,1,1])
    next_index = tf.where(closest_index == 0, tf.constant(1,dtype=closest_index.dtype), closest_index-1)
    next_index = tf.where(next_index >= N-2,
                          N-2, 
                          closest_index-tf.constant(1,dtype=closest_index.dtype))
    aug_next_index = tf.stack([
                            tf.tile(tf.range(M,dtype=tf.int64)[:,None],[1,B]),
                            tf.tile(tf.range(B,dtype=tf.int64)[None,:],[M,1]),
                            next_index
                        ],-1)  #MxBx2
    pts_n = tf.gather_nd(manifold_points_e, aug_next_index)
    pts_e = tf.gather_nd(manifold_points_e2, aug_closest_index)
    return pts_e, pts_n

def batched_closest_point_to_manifold_v2(manifold_points, test_points):
    """
    Here we consider manifold points, which are a BxNx2 tensor, representing a batch of B
    manifolds on a 2D space.
    """
    #find the closest points by euclidean distance, an MxB tensor
    z = batched_closest_and_next_point_on_manifold(manifold_points, test_points)
    closest_manifold_points = z[0]
    next_manifold_points = z[1]
    e2s = next_manifold_points - closest_manifold_points #MxBx2
    e1s = test_points[:,None,:] - closest_manifold_points #MxBx2
    e1len_squared = tf.reduce_sum(e1s*e1s,-1)
    e2len_squared = tf.reduce_sum(e2s*e2s,-1)
    e2len = tf.math.sqrt(e2len_squared)
    projected_on_line = (tf.reduce_sum(e1s*e2s,-1)/e2len_squared)[:,:,None]*e2s
    ps = projected_on_line + closest_manifold_points
    return ps

def batched_closest_euclidean_to_manifold_v2(manifold_points, test_points):
    """
    Here we consider manifold points, which are a BxNx2 tensor, representing a batch of B
    manifolds on a 2D space.
    """
    z = batched_closest_and_next_point_on_manifold(manifold_points, test_points)
    closest_manifold_points = z[0]
    next_manifold_points = z[1]
    return closest_manifold_points

def batched_sqdist_to_closest_point_on_manifold_v2(manifold_points, test_points):
    """
    Here we consider manifold points, which are a BxNx2 tensor, representing a batch of B
    manifolds on a 2D space.
    """
    ps = batched_closest_point_to_manifold_v2(manifold_points, test_points)
    sqdist2test = tf.reduce_sum(tf.math.square(test_points[:,None,:] - ps),-1)
#     sqeuclidean2test = tf.reduce_sum(tf.math.square(test_points[:,None,:] - closest_manifold_points),-1)
    return sqdist2test


# -

class TwoLineFit(gpflow.base.Module, gpflow.models.InternalDataTrainingLossMixin):
    r"""
    Gaussian Process Regression.
    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood of this model is sometimes referred to as the 'log
    marginal likelihood', and is given by
    .. math::
       \log p(\mathbf y \,|\, \mathbf f) =
            \mathcal N(\mathbf{y} \,|\, 0, \mathbf{K} + \sigma_n \mathbf{I})
    """
    def __init__(
        self,
        user_line1,
        user_line2,
        energy1=None,
        energy2=None,
        a=None, 
        b=None, 
        c=None,
        p=None, 
        q=None, 
        r=None,
        t=None, 
        u=None, 
        v=None,
        z=None,
        radius=None,
        mm_to_pixels = 20,
        pts_per_iteration = 1000,
        theta_min = -np.pi/8,
        theta_max = np.pi/8,
        xtal = CrystalGe(4,4,0)
    ):
        super().__init__()
        
        if energy1 is None:
            raise ValueError('must specify initial energy1')
        if energy2 is None:
            raise ValueError('must specify initial energy2')
        if a is None:
            raise ValueError('must specify initial a')
        if b is None:
            raise ValueError('must specify initial b')
        if c is None:
            raise ValueError('must specify initial c')
        if p is None:
            raise ValueError('must specify initial p')
        if q is None:
            raise ValueError('must specify initial q')
        if r is None:
            raise ValueError('must specify initial r')
        if t is None:
            raise ValueError('must specify initial t')
        if u is None:
            raise ValueError('must specify initial u')
        if v is None:
            raise ValueError('must specify initial v')    
        if radius is None:
            raise ValueError('must specify the xtal bending radius')
            
        parallel_models = np.atleast_1d(a).shape[0]
            
        self.energy1 = tf.constant(float(energy1),dtype=tf.float64)#gpflow.Parameter(float(energy1))
        self.energy2 = tf.constant(float(energy2),dtype=tf.float64)#gpflow.Parameter(float(energy2))
        self.a = gpflow.Parameter(np.atleast_1d(a)[:,None])
        self.b = gpflow.Parameter(np.atleast_1d(b)[:,None])
        self.c = gpflow.Parameter(np.atleast_1d(c)[:,None])
        self.p = gpflow.Parameter(np.atleast_1d(p)[:,None])
        self.q = gpflow.Parameter(np.atleast_1d(q)[:,None])
        self.r = gpflow.Parameter(np.atleast_1d(r)[:,None])
        self.t = gpflow.Parameter(np.atleast_1d(t)[:,None], transform=tfb.Exp())
        self.u = gpflow.Parameter(np.atleast_1d(u)[:,None])
        self.v = gpflow.Parameter(np.atleast_1d(v)[:,None])
        self.radius = radius
        self.xtal = xtal
        self.mm_to_pixels = mm_to_pixels  #convert from mm to det pixel space (20pixels/mm)
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.theta_range = tf.cast(tf.linspace(self.theta_min,
                                       self.theta_max,
                                       pts_per_iteration),tf.float64)[None,:]
        self.user_data1 = tf.convert_to_tensor(user_line1)
        self.user_data2 = tf.convert_to_tensor(user_line2)
    
    def nan_filter(self, energy):
        y = dety(self.xtal(energy),self.a,self.b,self.c,self.p,self.q,self.r,
                                 self.t,self.u,self.v,self.radius,self.theta_range)
        z = detz(self.xtal(energy),self.a,self.b,self.c,self.p,self.q,self.r,
                                 self.t,self.u,self.v,self.radius,self.theta_range)
        nan_filter_y = tf.math.logical_not(tf.math.is_nan(y))
        nan_filter_z = tf.math.logical_not(tf.math.is_nan(z))
        nan_filter = tf.math.logical_and(nan_filter_y, nan_filter_z)
        # to prevent ragged filtering, we impose the same filter over all batches
        return tf.reduce_all(nan_filter, 0)
    
    def raytracing_pts(self, energy) -> tf.Tensor:   #call realspace_raytracing_predictions
        nan_filter = tf.stop_gradient(self.nan_filter(energy))
        T = tf.squeeze(self.theta_range)[nan_filter][None,:]
        
        y = dety(self.xtal(energy),self.a,self.b,self.c,self.p,self.q,self.r,
                                 self.t,self.u,self.v,self.radius,T)
        z = detz(self.xtal(energy),self.a,self.b,self.c,self.p,self.q,self.r,
                                 self.t,self.u,self.v,self.radius,T)
        detector_pts = tf.stack([y, z],-1)  # BxNonex2
        return detector_pts
    
    def raytracing_slice(self, energy, N: int):
        y = dety(self.xtal(energy),self.a[:N,:],self.b[:N,:],self.c[:N,:],self.p[:N,:],self.q[:N,:],self.r[:N,:],
                                 self.t[:N,:],self.u[:N,:],self.v[:N,:],self.radius,self.theta_range)
        z = detz(self.xtal(energy),self.a[:N,:],self.b[:N,:],self.c[:N,:],self.p[:N,:],self.q[:N,:],self.r[:N,:],
                                 self.t[:N,:],self.u[:N,:],self.v[:N,:],self.radius,self.theta_range)
        nan_filter_y = tf.math.logical_not(tf.math.is_nan(y))
        nan_filter_z = tf.math.logical_not(tf.math.is_nan(z))
        nan_filter = tf.math.logical_and(nan_filter_y, nan_filter_z)
        nan_filter = tf.reduce_all(nan_filter, 0)
        T = tf.squeeze(self.theta_range)[nan_filter][None,:]
        
        y = dety(self.xtal(energy),self.a[:N,:],self.b[:N,:],self.c[:N,:],self.p[:N,:],self.q[:N,:],self.r[:N,:],
                                 self.t[:N,:],self.u[:N,:],self.v[:N,:],self.radius,T)
        z = detz(self.xtal(energy),self.a[:N,:],self.b[:N,:],self.c[:N,:],self.p[:N,:],self.q[:N,:],self.r[:N,:],
                                 self.t[:N,:],self.u[:N,:],self.v[:N,:],self.radius,T)
        detector_pts = tf.stack([y, z],-1)  # BxNonex2
        return detector_pts


    @staticmethod
    def minimum_2D_distance(manifold_points,test_points):
        dists = tf.math.sqrt(tf.clip_by_value(batched_sqdist_to_closest_point_on_manifold_v2(manifold_points, test_points),1E-6,1E6))
        return tf.reduce_mean(dists,axis=0)
    
    def _individual_training_losses(self):
        loss1 = self.minimum_2D_distance(self.raytracing_pts(self.energy1)*self.mm_to_pixels, self.user_data1)
        loss2 = self.minimum_2D_distance(self.raytracing_pts(self.energy2)*self.mm_to_pixels, self.user_data2) 
        return loss1 + loss2
    
    def get_params(self, index):
        p1 = {
            'a': self.a[index].numpy().item(),
            'b': self.b[index].numpy().item(),
            'c': self.c[index].numpy().item(),
            'p': self.p[index].numpy().item(),
            'q': self.q[index].numpy().item(),
            'r': self.r[index].numpy().item(),
            't': self.t[index].numpy().item(),
            'u': self.u[index].numpy().item(),
            'v': self.v[index].numpy().item(),
            'energy': self.energy1,
            'radius': self.radius,
            'theta_min': self.theta_min,
            'theta_max': self.theta_max,
            'mm_to_pixels': self.mm_to_pixels
        }
        p2 = {
            'a': self.a[index].numpy().item(),
            'b': self.b[index].numpy().item(),
            'c': self.c[index].numpy().item(),
            'p': self.p[index].numpy().item(),
            'q': self.q[index].numpy().item(),
            'r': self.r[index].numpy().item(),
            't': self.t[index].numpy().item(),
            'u': self.u[index].numpy().item(),
            'v': self.v[index].numpy().item(),
            'energy': self.energy2,
            'radius': self.radius,
            'theta_min': self.theta_min,
            'theta_max': self.theta_max,
            'mm_to_pixels': self.mm_to_pixels
        }
        return p1, p2
        
    
    def _training_loss(self):
        loss1 = self.minimum_2D_distance(self.raytracing_pts(self.energy1)*self.mm_to_pixels, self.user_data1)
        loss2 = self.minimum_2D_distance(self.raytracing_pts(self.energy2)*self.mm_to_pixels, self.user_data2) 
        return tf.reduce_mean(loss1 + loss2)
