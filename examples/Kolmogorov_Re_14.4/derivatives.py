#!/usr/bin/env python3

import numpy as np
import h5py
import scipy.integrate as integrate


class kolmo_stuff:
    def __init__(self,om,Re=14.4):
        aal = 2/3 #AntiAliasing
        self.im = 0 +1j

        n = 2


        self.shape = om.shape
        y = np.linspace(0,2*np.pi,self.shape[2],endpoint=False)


        om_fft = np.fft.rfft2(np.real(om))
        self.fft_shape = om_fft.shape
        self.Re = Re
        self.F = (-n * np.cos(n*y))[None,None,:]
        self.kx = np.fft.fftfreq(self.shape[1],d=1/self.shape[1])[None,:,None]
        self.ky = np.fft.rfftfreq(self.shape[2],d=1/self.shape[2])[None,None,:]

        kx_mask = (np.abs(self.kx) <= aal*np.max(np.abs(self.kx)))
        ky_mask = (np.abs(self.ky) <= aal*np.max(np.abs(self.ky)))

        self.mask = kx_mask * ky_mask
        self.norm = self.kx**2 + self.ky**2
        self.norm[:,0,0] = 1.0
        
    def vort_to_vel(self,om_fft):
        stream_fft = - om_fft/self.norm
        stream_fft[:,0,0] = 0.0 #galilean
        u_fft = - self.im * stream_fft * self.ky
        v_fft = self.im * stream_fft * self.kx
        return (u_fft,v_fft)
        

    def get_deriv_ts(self,om_t):
        om_fft = np.fft.rfft2(om_t,axes=(1,2))

        u_fft, v_fft = self.vort_to_vel(om_fft) 
        u = np.fft.irfft2(u_fft*self.mask,axes=(1,2))
        v = np.fft.irfft2(v_fft*self.mask,axes=(1,2))
        lapl_om = np.fft.irfft2(-self.kx**2 * om_fft*self.mask,axes=(1,2)) + np.fft.irfft2(-self.ky**2 * om_fft*self.mask,axes=(1,2)) 
        om_x =  np.fft.irfft2(self.im * self.kx * om_fft * self.mask,axes=(1,2))
        om_y =  np.fft.irfft2(self.im * self.ky * om_fft * self.mask,axes=(1,2))

        p1 = u * om_x + v * om_y

        res = lapl_om/self.Re - p1

        return(res + self.F)
    def get_deriv_x_ts(self,om_t):
        om_x =  np.fft.irfft2(self.im * self.kx * np.fft.rfft2(om_t,axes=(1,2)) * self.mask,axes=(1,2))
        return(-1 * om_x)
