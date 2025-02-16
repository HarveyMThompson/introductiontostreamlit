# v2st_rosenbrock_2DNMSimplex.py 
# Python implementation of the Nelder Mead Simplex algorithm for use in streamlit
import numpy as np
import streamlit as st
import streamlit_scrollable_textbox as stx
import matplotlib.pyplot as plt
import sys
import sympy as sp
import time

from NMsimplex import *
from scipy import optimize

###############################################################################
# Functions
###############################################################################
# objective function
def obj2(x):
    x0 = x[0]
    x1 = x[1]
    obj2 = obj(x0,x1)
    return obj2

# define 2D Nelder-Mead Simplex program
def nelder_mead():

    x_0 = st.session_state.x_0
    y_0 = st.session_state.y_0
    xinitial = [x_0, y_0]
    
    res = optimize.minimize(obj2,xinitial)
    print('Global minimum value ={0:6.3f} at x = {1:6.3f} {2:6.3f}\n'.format(res.fun,res.x[0],res.x[1]))

    # initial NM Simplex setup parameters
    ndv=2
    x0=np.array([x_0,y_0])
    c = st.session_state.c
    alpha = st.session_state.alpha
    gamma = st.session_state.gamma
    beta = st.session_state.beta
    rho = st.session_state.rho
    tol = st.session_state.tol   # convergence tolerance

    converged = 0; maxiters = 100;
    # store results of simplex calculations
    x_simplex = np.zeros((maxiters+1,3,2),dtype='float64')
    fmin_simplex = np.zeros(maxiters+1,dtype='float64')
    xp = np.zeros((3,2),dtype='float64'); fp = np.zeros(3,dtype='float64');   # arrays for determining next simplex

    # create initial NM simplex
    xL,xM,xH = create_simplex(x0,c,obj2)
    niter = 0
    outstr = ''
    outstr+=('Nelder-Mead Simplex Parameters:\n')
    outstr+=('-------------------------------\n')
    outstr+=('x0={0:8.3f} {1:8.3f}, f0={2:8.3f}\n'.format(x0[0],x0[1],obj2(x0)))
    outstr+=('c={0:4.2f} alpha={1:4.2f} gamma={2:4.2f} beta={3:4.2f} rho={4:4.2f} tol={5:8.5f} \n'.
              format(c,alpha,gamma,beta,rho,tol))

    outstr+=('Initial simplex\n')
    outstr+=('xL={0:8.3f} {1:8.3f}, fL={2:8.3e}\n'.format(xL[0],xL[1],obj2(xL)))
    outstr+=('xM={0:.3f} {1:8.3f}, fM={2:8.3e}\n'.format(xM[0],xM[1],obj2(xM)))
    outstr+=('xL={0:8.3f} {1:8.3f}, fL={2:8.3e}\n'.format(xH[0],xH[1],obj2(xH)))
    x_simplex[0][0][:]=xL; x_simplex[0][1][:]=xM; x_simplex[0][2][:]=xH;
    fmin_simplex[0]=obj2(xL)

    while (converged == 0):
    
        niter = niter+1
        xCE = 0.5*(xL + xM)
    
        # Carry out reflection operation
        xR,fR = reflection(xL,xM,xH,alpha,obj2)
        fL=obj2(xL); fM=obj2(xM); fH=obj2(xH);
    
        if ((fL < fR) and (fR < fM)):
        
            # update the simplex points
            xH = xM
            xM = xR
            xCE = 0.5*(xL + xM)
            converged = check_convergence(xL,xM,xH,xCE,tol,niter,maxiters,obj2)
            outstr+=('Reflection carried out for iteration {0:3d}\n'.format(niter))
        
        elif (fR < fL):
        
            # carry out expansion
            xE,fE = expansion(xCE,xR,gamma,obj2)
            oldxL = xL; oldxM = xM; oldxH = xH;
        
            if (fE < fR):
                xL = xE
                xM = oldxL
                xH = oldxM
            else:
                xL = xR
                xM = oldxL
                xH = oldxM
        
            xCE = 0.5*(xL + xM)
            converged = check_convergence(xL,xM,xH,xCE,tol,niter,maxiters,obj2)
            outstr+=('Expansion carried out for iteration {0:3d}\n'.format(niter))
       
        elif ((fR > fM) and (fR < fH)):
        
            # carry out outside contraction
            xOC,fOC = outside_contraction(xCE,xR,beta,obj2)
        
            if (fOC < fR):
            
                # update simplex points based on three smallest function values
                fp[0]=fL; fp[1]=fM; fp[2]=fOC
                xp = np.zeros((3,2),dtype='float64') 
                xp[0,:]=xL; xp[1,:]=xM; xp[2,:]=xOC
            
                # update simplex with three smallest function values
                ipos = np.argsort(fp)
                xL = xp[ipos[0],:]
                xM = xp[ipos[1],:]
                xH = xp[ipos[2],:]
            
            else:
            
                # carry out shrinking operation
                newxM,newxH = shrinking(xL,xM,xH,rho,obj2)
                xM = newxM
                xH = newxH
            
        
            xCE = 0.5*(xL + xM)
            converged = check_convergence(xL,xM,xH,xCE,tol,niter,maxiters,obj2)
            outstr+=('Outside contraction carried out for iteration {0:3d}\n'.format(niter))
        
        elif (fR > fH):
        
            # carry out inside contraction
            xIC,fIC = inside_contraction(xCE,xR,beta,obj2)
        
            if (fIC < fH):
            
                # update simplex points based on three smallest function values
                fp[0]=fL; fp[1]=fM; fp[2]=fIC
                xp = np.zeros((3,2),dtype='float64')                 
                xp[0,:]=xL; xp[1,:]=xM; xp[2,:]=xIC
            
                # update simplex with three smallest function values
                ipos = np.argsort(fp)
                xL = xp[ipos[0],:]
                xM = xp[ipos[1],:]
                xH = xp[ipos[2],:]
            
            else:
            
                # carry out shrinking operation
                newxM,newxH = shrinking(xL,xM,xH,rho,obj2)
                xM = newxM
                xH = newxH
            
            xCE = 0.5*(xL + xM)
            converged = check_convergence(xL,xM,xH,xCE,tol,niter,maxiters,obj2)
            outstr+=('Inside contraction carried out for iteration {0:3d}\n'.format(niter))
        
        outstr+=('After {0:3d} iterations, simplex is given by:\n'.format(niter))
        outstr+=('xL={0:8.3f} {1:8.3f}, fL={2:8.3e}\n'.format(xL[0],xL[1],obj2(xL)))
        outstr+=('xM={0:8.3f} {1:8.3f}, fM={2:8.3e}\n'.format(xM[0],xM[1],obj2(xM)))
        outstr+=('xH={0:8.3f} {1:8.3f}, fH={2:8.3e}\n'.format(xH[0],xH[1],obj2(xH)))
        x_simplex[niter][0][:]=xL; x_simplex[niter][1][:]=xM; x_simplex[niter][2][:]=xH;
        fmin_simplex[niter]=obj2(xL)

    # end while loop

    outstr+=('NM simplex converged with tol={0:6.3e} after {1:3d} iterations\n'.format(tol,niter))
    outstr+=('Minimum f={0:8.3e} at x= {1:8.3e} {2:8.3e}\n'.format(obj2(xL),xL[0],xL[1]))

    st.session_state.Calcs=outstr
    st.session_state.x_simplex=x_simplex
    st.session_state.fmin_simplex=fmin_simplex
    st.session_state.niter=niter
    st.session_state.xMin=x_simplex[niter][0][0]
    st.session_state.yMin=x_simplex[niter][0][1]
    st.session_state.fMin=fmin_simplex[niter]

    print('finished Nelder-Mead')

# Function to create and update the plot for Nelder Mead Simplex
def create_plot():

    # Run the Nelder-Mead Simplex
    nelder_mead()
    x_simplex=st.session_state.x_simplex
    fmin_simplex=st.session_state.fmin_simplex
    niter=st.session_state.niter

    # create initial NM simplex
    x_0 = st.session_state.x_0
    y_0 = st.session_state.y_0
    x0=np.array([x_0,y_0])
    c = st.session_state.c
    xL,xM,xH = create_simplex(x0,c,obj2)

    plt.figure()    
    # now plot out solution
    X, Y = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-1, 3, 101))
    # Z = 100*((Y-X**2)**2)+(1-X)**2
    Z = obj(X,Y)

    # plot out first simplex
    xL[0]=x_simplex[0][0][0]; xL[1]=x_simplex[0][0][1]; 
    xM[0]=x_simplex[0][1][0]; xM[1]=x_simplex[0][1][1]; 
    xH[0]=x_simplex[0][2][0]; xH[1]=x_simplex[0][2][1]; 
    initial_simplex=np.zeros((4,3),dtype='float64')
    initial_simplex[0][0]=xL[0]; initial_simplex[0][1]=xL[1]
    initial_simplex[1][0]=xM[0]; initial_simplex[1][1]=xM[1]
    initial_simplex[2][0]=xH[0]; initial_simplex[2][1]=xH[1]
    initial_simplex[3][0]=xL[0]; initial_simplex[3][1]=xL[1]
    plt.plot(initial_simplex[:,0],initial_simplex[:,1],'r-')

    # plot out second simplex
    xL[0]=x_simplex[1][0][0]; xL[1]=x_simplex[1][0][1]; 
    xM[0]=x_simplex[1][1][0]; xM[1]=x_simplex[1][1][1]; 
    xH[0]=x_simplex[1][2][0]; xH[1]=x_simplex[1][2][1]; 
    initial_simplex=np.zeros((4,3),dtype='float64')
    initial_simplex[0][0]=xL[0]; initial_simplex[0][1]=xL[1]
    initial_simplex[1][0]=xM[0]; initial_simplex[1][1]=xM[1]
    initial_simplex[2][0]=xH[0]; initial_simplex[2][1]=xH[1]
    initial_simplex[3][0]=xL[0]; initial_simplex[3][1]=xL[1]
    plt.plot(initial_simplex[:,0],initial_simplex[:,1],'b-')

    # plot out third simplex
    xL[0]=x_simplex[2][0][0]; xL[1]=x_simplex[2][0][1]; 
    xM[0]=x_simplex[2][1][0]; xM[1]=x_simplex[2][1][1]; 
    xH[0]=x_simplex[2][2][0]; xH[1]=x_simplex[2][2][1]; 
    initial_simplex=np.zeros((4,3),dtype='float64')
    initial_simplex[0][0]=xL[0]; initial_simplex[0][1]=xL[1]
    initial_simplex[1][0]=xM[0]; initial_simplex[1][1]=xM[1]
    initial_simplex[2][0]=xH[0]; initial_simplex[2][1]=xH[1]
    initial_simplex[3][0]=xL[0]; initial_simplex[3][1]=xL[1]
    plt.plot(initial_simplex[:,0],initial_simplex[:,1],'k-')

    # plot out contours
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('Nelder Mead search on '+st.session_state.expression)
    # plot1.set_title('f=$100(y-x^2)^2+(1-x)^2$')
    CS = plt.contour(X, Y, Z, levels=[0, 2.5, 25, 100])
    xout=np.linspace(0,1,2); yout = np.linspace(0,1,2)
    for i in range(niter-1):
        xout[0]=x_simplex[i][0][0]; yout[0]=x_simplex[i][0][1]; 
        xout[1]=x_simplex[i+1][0][0]; yout[1]=x_simplex[i+1][0][1]; 
        plt.scatter(xout[0],yout[0],c='k',marker='o')
        plt.plot(xout,yout,'k-')

    plt.scatter(xout[1],yout[1],c='r',marker='o',)

    return plt
   
# Function to create plots for Nelder Mead Simplex used in animation
def animate_plot():

    # Run the Nelder-Mead Simplex
    nelder_mead()
    x_simplex=st.session_state.x_simplex
    fmin_simplex=st.session_state.fmin_simplex
    niter=st.session_state.niter

    # create initial NM simplex
    x_0 = st.session_state.x_0
    y_0 = st.session_state.y_0
    x0=np.array([x_0,y_0])
    c = st.session_state.c
    xL,xM,xH = create_simplex(x0,c,obj2)

    plt.figure()    
    # now plot out solution
    X, Y = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-1, 3, 101))
    # Z = 100*((Y-X**2)**2)+(1-X)**2
    Z = obj(X,Y)

    # plot out first simplex
    xL[0]=x_simplex[0][0][0]; xL[1]=x_simplex[0][0][1]; 
    xM[0]=x_simplex[0][1][0]; xM[1]=x_simplex[0][1][1]; 
    xH[0]=x_simplex[0][2][0]; xH[1]=x_simplex[0][2][1]; 
    initial_simplex=np.zeros((4,3),dtype='float64')
    initial_simplex[0][0]=xL[0]; initial_simplex[0][1]=xL[1]
    initial_simplex[1][0]=xM[0]; initial_simplex[1][1]=xM[1]
    initial_simplex[2][0]=xH[0]; initial_simplex[2][1]=xH[1]
    initial_simplex[3][0]=xL[0]; initial_simplex[3][1]=xL[1]
    plt.plot(initial_simplex[:,0],initial_simplex[:,1],'r-')

    # plot out second simplex
    xL[0]=x_simplex[1][0][0]; xL[1]=x_simplex[1][0][1]; 
    xM[0]=x_simplex[1][1][0]; xM[1]=x_simplex[1][1][1]; 
    xH[0]=x_simplex[1][2][0]; xH[1]=x_simplex[1][2][1]; 
    initial_simplex=np.zeros((4,3),dtype='float64')
    initial_simplex[0][0]=xL[0]; initial_simplex[0][1]=xL[1]
    initial_simplex[1][0]=xM[0]; initial_simplex[1][1]=xM[1]
    initial_simplex[2][0]=xH[0]; initial_simplex[2][1]=xH[1]
    initial_simplex[3][0]=xL[0]; initial_simplex[3][1]=xL[1]
    plt.plot(initial_simplex[:,0],initial_simplex[:,1],'b-')

    # plot out third simplex
    xL[0]=x_simplex[2][0][0]; xL[1]=x_simplex[2][0][1]; 
    xM[0]=x_simplex[2][1][0]; xM[1]=x_simplex[2][1][1]; 
    xH[0]=x_simplex[2][2][0]; xH[1]=x_simplex[2][2][1]; 
    initial_simplex=np.zeros((4,3),dtype='float64')
    initial_simplex[0][0]=xL[0]; initial_simplex[0][1]=xL[1]
    initial_simplex[1][0]=xM[0]; initial_simplex[1][1]=xM[1]
    initial_simplex[2][0]=xH[0]; initial_simplex[2][1]=xH[1]
    initial_simplex[3][0]=xL[0]; initial_simplex[3][1]=xL[1]
    plt.plot(initial_simplex[:,0],initial_simplex[:,1],'k-')

    # plot out contours
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('Nelder Mead search on '+st.session_state.expression)
    # plot1.set_title('f=$100(y-x^2)^2+(1-x)^2$')
    CS = plt.contour(X, Y, Z, levels=[0, 2.5, 25, 100])
    xout=np.linspace(0,1,2); yout = np.linspace(0,1,2)
    for i in range(st.session_state.ncalls):
        xout[0]=x_simplex[i][0][0]; yout[0]=x_simplex[i][0][1]; 
        xout[1]=x_simplex[i+1][0][0]; yout[1]=x_simplex[i+1][0][1]; 
        plt.scatter(xout[0],yout[0],c='k',marker='o')
        plt.plot(xout,yout,'k-')

    if (st.session_state.ncalls == st.session_state.niter - 2):
        plt.scatter(xout[1],yout[1],c='r',marker='o',)

    return plt

     
###############################################################################
# Main program
###############################################################################

# Streamlit commands
st.title("Nelder-Mead Simplex Algorithm Program")
st.write("This application enables you to experiment with Nelder-Mead solution of 2-D equations")

# initialise the objective function
exp = st.text_input("Objective Function",value='100*(y-x^2)^2 + (1-x)^2')
if exp is not None:
    expression = exp
else:
    expression = '100*(y-x^2)^2 + (1-x)^2'

x = sp.symbols('x')
y = sp.symbols('y')

try:
    obj_function = sp.sympify(expression)
except Exception as e:
    st.error(f"An error occured using sympify: str{e}")
    st.stop()
    
try:
    obj = sp.lambdify([x,y], obj_function, 'numpy')
except Exception as e:
    st.error(f"An error occured using lambdify: str{e}")
    st.stop()
    
st.session_state.obj = obj
st.session_state.expression = expression

st.write("Parameter settings")

# First row
cols = st.columns([1,1,1,1,1])

alpha = cols[0].number_input(chr(945),0.1,5.0,1.0)
st.session_state.alpha = alpha

beta = cols[1].number_input(chr(946),0.1,5.0,0.5,0.1)
st.session_state.beta = beta

gamma = cols[2].number_input(chr(947),0.1,5.0,2.0,0.1)
st.session_state.gamma = gamma

rho = cols[3].number_input(chr(961),0.1,5.0,0.5,0.1)
st.session_state.rho = rho

tol = cols[4].number_input("Tolerance",0.00001,0.01,0.0001)
st.session_state.tol = tol

# Second row
cols2 = st.columns([1,1,1,1,1])

x_0 = cols2[0].number_input("$x_0$",-5.0,5.0,-1.2,0.1)
st.session_state.x_0 = x_0

y_0 = cols2[1].number_input("$y_0$",-5.0,5.0,1.0,0.1)
st.session_state.y_0= y_0

c = cols2[2].number_input("c",0.1,5.0,2.0,0.1)
st.session_state.c = c

# Get initial solution of nelder mead simplex calculations
cols3 = st.columns([1,1,1])
nelder_mead()
cols3[0].text("x min="+"{:.4f}".format(st.session_state.xMin))
cols3[1].text("y min="+"{:.4f}".format(st.session_state.yMin))
cols3[2].text("f min="+"{:.4e}".format(st.session_state.fMin))

if 'plotgraph' not in st.session_state:
    st.session_state.plotgraph = 1
    if st.button("Plot Graph and Calculations"):
        figplt = create_plot()
        st.pyplot(figplt)
        st.write('Nelder Mead Calculations')
        stx.scrollableTextbox(st.session_state.Calcs,height=200)
else:
    figplt = create_plot()
    st.pyplot(figplt)
    st.write('Nelder Mead Calculations')
    stx.scrollableTextbox(st.session_state.Calcs,height=200)

    # animate calculations
    if st.button("Animate Graph"):
        st.session_state.ncalls = 0
        figplt = animate_plot()
        the_plot = st.pyplot(figplt)
        for i in range(st.session_state.niter - 2):
            st.session_state.ncalls += 1
            time.sleep(0.1)
            figplt = animate_plot()
            the_plot.pyplot(figplt)
    