# NMsimplex.py
import numpy as np

# create and sort initial simplex points
def create_simplex(x0,c,obj):
    
    xp_ini = np.zeros((3,2),dtype='float64'); fp_ini = np.zeros(3,dtype='float64');
    xp_ini[0][0] = x0[0]; xp_ini[0][1] = x0[1]; fp_ini[0] = obj(xp_ini[0,:]);
    xp_ini[1][0] = xp_ini[0][0] + 0.96593*c; xp_ini[1][1] = xp_ini[0][1] + 0.25882*c;
    fp_ini[1] = obj(xp_ini[1,:])
    xp_ini[2][0] = xp_ini[0][0] + 0.25882*c; xp_ini[2][1] = xp_ini[0][1] + 0.96593*c;
    fp_ini[2] = obj(xp_ini[2,:])
    ipos = np.argsort(fp_ini)
    xL = xp_ini[ipos[0],:]
    xM = xp_ini[ipos[1],:]
    xH = xp_ini[ipos[2],:]
    
    return xL,xM,xH

# Reflection operation
def reflection(xL,xM,xH,alpha,obj):
    
    # calculate centroid excluding worst point (biggest function value)
    xCE = 0.5*(xL + xM)
    
    # calculate reflection point
    xR = (1+alpha)*xCE - alpha*xH
    fR = obj(xR)
    
    return xR,fR

# end reflection

# Expansion operation
def expansion(xCE,xR,gamma,obj):
    
    # calculate expansion point
    xE = gamma*xR + (1-gamma)*xCE
    fE = obj(xE)
    
    return xE,fE

# end expansion

# Outside Contraction operation
def outside_contraction(xCE,xR,beta,obj):
    
    # calculate contraction point
    xOC = xCE + beta*(xR-xCE)
    fOC = obj(xOC)
    
    return xOC,fOC

# end outside contraction

# Inside Contraction operation
def inside_contraction(xCE,xR,beta,obj):
    
    # calculate contraction point
    xIC = xCE - beta*(xR-xCE)
    fIC = obj(xIC)
    
    return xIC,fIC

# end inside contraction

# Shrinking operation
def shrinking(xL,xM,xH,rho,obj):
    
    # calculate shrinkage points about xL
    newxM = xL + rho*(xM-xL)
    newxH = xL + rho*(xH-xL)
    
    return newxM,newxH

# end shrinking

# check convergence
def check_convergence(xL,xM,xH,xCE,tol,niter,maxiters,obj):
    
    std = (obj(xL)-obj(xCE))**2 + (obj(xM)-obj(xCE))**2 + (obj(xH)-obj(xCE))**2
    std=np.sqrt(std)/3.0
    converged = 0
    if ((std < tol) or (niter==(maxiters-1))):
        converged = 1
        
    return converged

# end check_convergence