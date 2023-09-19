#!/usr/bin/env python

import numpy as np
from numpy.linalg import inv
from numpy.random import uniform, standard_cauchy

def pe_distance(pos1,pos2,box_size=500.):
    '''Calculate periodic distance between pos1 and pos2. 
    pos1: [[x, y, z], [...], [...]] or [x, y, z]
    pos2: [[x, y, z], [...], [...]] or [x, y, z]
    box_size: The size of periodic box
    
    Return:
    np array of periodic distances 
    '''
    if not hasattr(pos1, '__array__'):
        pos1 = np.array(pos1)
    if not hasattr(pos2, '__array__'):
        pos2 = np.array(pos2)
    if len(pos1.shape)>len(pos2.shape):
        d = np.abs(pos1-pos2)
    else:
        d = np.abs(pos2-pos1)
    d[d>box_size/2.] = d[d>box_size/2.]-box_size
    return np.sqrt(np.sum(d**2.,axis=1))
    
def pe_distance1d(pos1,pos2,box_size=500.):
    '''
    Calculate 1-dimensional periodic distance of pos1 with respect to pos2. 
    pos1: 1-d positions
    pos2: 1-d positions
    box_size: The size of periodic box
    
    Return:
    np array of periodic distances 
    '''
    if not hasattr(pos1, '__array__'):
        pos1 = np.array(pos1)
    if not hasattr(pos2, '__array__'):
        pos2 = np.array(pos2)
    d = pos1-pos2
    d[d>box_size/2] = d[d>box_size/2]-box_size
    d[d<-box_size/2] = d[d<-box_size/2]+box_size
    return d

def angle(x,y,z):
    '''z=r cos theta, y = x tan phi
    return theta, phi
    '''
    r = np.sqrt(x**2+y**2+z**2)
    return np.arccos(z/r), np.arctan2(y,x)
    
def remove_perh(x, h):
    '''convert Mpc/h to Mpc and so on'''
    return x/h
    
def add_perh(x, h):
    '''convert Mpc to Mpc/h and so on'''
    return x*h
    
def physical_2_comoving(physical = None, comoving=None, aexp = 1.0):
    '''
    Usage: 
    physical_2_comoving(physical=x,aexp=0.5) to convert physical x Mpc to comoving at a=0.5
    physical_2_comoving(comoving=x,aexp=0.5) to convert comoving x Mpc to physical at a=0.5
    '''
    if comoving==None:
        return physical/aexp
    else:
        return comoving*aexp
        
    
#def vector_xyz_rt(pos,vec):
#    vecr = np.array(pos[0]*vec[0]+pos[1]*vec[1]+pos[2]*vec[2])/np.sqrt(pos[0]**2+pos[1]**2+pos[2]**2)
#    return vecr
    
def vector_xyz_rt(pos,vec):
    '''Convert vec=[vx,vy,vz] to [vr,vtheta,vphi] relative to 0 point at pos=[x,y,z]'''
    theta, phi = angle(*pos)
    vecr = vec[0]*np.sin(theta)*np.cos(phi)+vec[1]*np.sin(theta)*np.sin(phi)+vec[2]*np.cos(theta)
    vect = vec[0]*np.cos(theta)*np.cos(phi)+vec[1]*np.cos(theta)*np.sin(phi)-vec[2]*np.sin(theta)
    vecp = -vec[0]*np.sin(phi) + vec[1]*np.cos(phi)
    return vecr, vect, vecp

def binmid(x,log=False):
    '''
    Return midpoints of x, arithmeticly or geometricly (log=True).
    '''
    if log:
        return np.sqrt(x[1:]*x[:-1])
    else:
        return (x[1:]+x[:-1])/2.

def diff(x,y,log=False):
    '''
    Return dy/dx or dlogy/dlogx (log=True). Call binmid for corresponding x.
    '''
    if log:
        return np.log(y[1:]/y[:-1])/np.log(x[1:]/x[:-1])
    else:
        return (y[1:]-y[:-1])/(x[1:]-x[:-1])

def find_bin(pts,bin,binsize='None'):
    '''
    Return the index of bin pt belongs to bin, where the returned index corresponds to binmid.
    If it is equal to bin edge, it will put in lower bin.
    If the bin is equally spaced linearly, binsize='linear' and logarithmically, binsize='log'
    for faster execution.
    '''
    if binsize=='linear':
        size = bin[1]-bin[0]
        pts = pts-bin[0]
        return np.floor(pts/size).astype('int')
    elif binsize=='log':
        size = np.log(bin[1]/bin[0])
        pts = np.log(pts/bin[0])
        return np.floor(pts/size).astype('int')
    else:
        if hasattr(pts, "__len__"):
            ptscopy = np.array([pts]*len(bin))
            bincopy = np.array([bin]*len(pts))
            return np.sum(ptscopy.T>bincopy,axis=1)-1        
        else:
            return np.sum(pts>bin)-1
         
def fit_matrix(x,y,cov=False):
    '''
    For 1D point data, fitting:
    x = np.vstack((np.ones(4),np.array([0.5011,0.6226,0.7441,1.000]))).T
    y = np.array([1.02621,1.02176,1.0136,0.99113])
    '''
    alpha = np.dot(x.T,x)
    beta = np.dot(x.T,y)
    a = np.dot(inv(alpha),beta)
    if cov:
        ybar = np.dot(x,a)
        dely = y-ybar
        ssq = np.dot(dely.T,dely)/(len(y)-len(a))
        cov_arr = np.sqrt(ssq*np.diag(inv(alpha)))
        return a, cov_arr
    else:
        return a
        
def lin_pol(x,pt1,pt2):
    '''
    Return linear interpolated result between pt1=[x1,y1] and pt2=[x2,y2] at x.
    '''
    return pt1[1]+(x-pt1[0])*(pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    
def exp_pol(x,pt1,pt2):
    '''
    Return exponential interpolated result between pt1=[x1,y1] and pt2=[x2,y2] at x.
    '''
    return pt1[1]*(pt2[1]/pt1[1])**((x-pt1[0])/(pt2[0]-pt1[0]))
    
def rand_dist(dist_func, func=None, lim=[-10,10], length=10000, args=None):
    '''
    Generate random number for any distribution function with rejection sampling
    within bounded limit. Useful if you don't know cumulative distribution function
    dist_func = distribution function used to generate random number
    func = fiducial function to be used. 
        If None, default is cauchy distribution at median=0, width=1
    lim = [lower, upper]
        Limit within which random number is generated. Infinite bound not acrepted. 
    length = approximate length of output.
    args = arguments for dist_func
    
    '''
    if func==None:
        func = lambda x: 1/(np.pi*(1+x**2))
    elif not callable(func):
        raise RuntimeError("You must input a callable func as a proposal distribution")
    test = np.linspace(lim[0],lim[1])
    if args==None:
        c = int(np.ceil(np.amax(dist_func(test)/func(test))))
    else:
        c = int(np.ceil(np.amax(dist_func(test,*args)/func(test))))
    u = uniform(size=length*c)
    y = standard_cauchy(size=length*c)
    limmask = (y<lim[1]) & (y>lim[0])
    y = y[limmask]
    u = u[limmask]
    gy = 1/(np.pi*(1+y**2))
    if args==None:
        fy = dist_func(y)
    else:
        fy = dist_func(y,*args)
    mask = u<fy/(c*gy)
    sel = y[mask]
    return sel
    
    
def crossmatch_pair(x1,y1,x2,y2):
    test = np.array((x1,y1)).T
    target = np.array((x2,y2)).T
    trial = (test[:,None]==target).all(2)
    ind1, ind2 = np.where(trial)
    return ind1, ind2


#astropy calculations   
from astropy import units as u
from astropy import constants as c
def vcirc(mass,redshift,mdef,cosmo):
    '''Calculate circular velocity in km/s for halos of mass M (Msun/h)'''
    rho_crit = cosmo.critical_density(redshift)
    if mdef == 'vir':
        x = cosmo.Om(redshift) - 1
        delta = 18 * np.pi**2 + 82.0 * x - 39.0 * x**2
        rho = delta*rho_crit
    elif mdef[-1] == 'c':
        delta = int(mdef[:-1])
        rho = delta*rho_crit
    elif mdef[-1] == 'm':
        delta = int(mdef[:-1])
        rho = delta*rho_crit*cosmo.Om(redshift)
    else:
        raise RuntimeError("Not correct mass definition")
    v = np.sqrt(c.G*(np.pi*4*rho/3)**(1./3)*(mass*u.Msun/cosmo.h)**(2./3))
    return v.to(u.km/u.s)
    
def delta_vir(z, cosmo):
    x = cosmo.Om(z) - 1
    return 18 * np.pi**2 + 82.0 * x - 39.0 * x**2
    
def virial_values(M, z, cosmo, mdef = 'vir', quantity='pressure', mu = 0.59):
    '''Calculate the virial/normalized values for different quantities of halos
    according to Lau+2015
    Parameters:
       M: halo mass in Msun unit, h=1
       z: redshift
       cosmo: astropy cosmology object
       mdef: mass definition, vir, xxc or xxm
       quantity: virial quantity to calculate, 
       radius, density (total), pressure, temperature, velocity 
       mu: mean particle weigh
    Return:
       virial quantity. Can use .to('xx') to convert to desired units.
       Default units are radius(kpc), P(keV/cm3), T(keV), v(km/s)
    '''
    rho_crit = cosmo.critical_density(z)
    if mdef == 'vir':
        x = cosmo.Om(z) - 1
        delta = 18 * np.pi**2 + 82.0 * x - 39.0 * x**2
        rho = delta*rho_crit
    elif mdef[-1] == 'c':
        delta = int(mdef[:-1])
        rho = delta*rho_crit
    elif mdef[-1] == 'm':
        delta = int(mdef[:-1])
        rho = delta*rho_crit*cosmo.Om(z)
    else:
        raise ValueError("Unsupported mdef")
    fb = cosmo.Ob0/cosmo.Om0
    M = M*u.Msun
    R = ((3*M/(np.pi*4*rho))**(1./3)).to('kpc')
    if quantity=='radius':
        return R
    elif quantity=='density':
        return rho    
    elif quantity=='pressure':
        return (fb*rho*c.G*M/(2*R)).to('keV/cm3')
    elif quantity=='temperature':
        return (c.G*M*mu*c.m_p/(2*R)).to('keV')
    elif quantity=='velocity':
        return np.sqrt(c.G*M/R).to('km/s')
    else:
        raise ValueError("Unsupported quantity")
    
#matplotlib stuff
import matplotlib.ticker as ticker

class MyLogFormatter(ticker.LogFormatter) :
    '''Usage: ax.xaxis.set_major_formatter(MyLogFormatter())
    '''
    def __call__(self,x,pos=None) :
        if (np.log10(x)) < 3 and (np.log10(x)) > -3 :
            return "$%g$" % (x,)
        else :
            return "$10^{%g}$" % (np.log10(x),)
