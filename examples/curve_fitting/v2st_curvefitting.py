# v2st_curvefitting.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.figure import Figure

##############################################################################
# Functions
##############################################################################
# Create function to carry out curve fitting
def fitting(xpts,ypts,x,order):
    p = np.polyfit(xpts,ypts,order)
    p_poly = np.poly1d(p)
    y_fit = p_poly(x)
    
    return y_fit

# read data to be used in plotting
def read_data():
    
    data = dict()
    
    builddata = open('build.txt','r').readlines()
    xpts = []
    ypts = []
    for line in builddata:
        xpts.append(float(line.split()[0]))
        ypts.append(float(line.split()[1]))    
    data['xpts']=xpts
    data['ypts']=ypts
    
    N = 101
    x = np.linspace(0,2,N)
    data['x']=x
    
    st.session_state.xpts = xpts
    st.session_state.ypts = ypts
    st.session_state.x = x
       
    return data

# Function to update the surrogate model plot
def update_curve_plot(data):

    # set data from dictionary
    xpts=st.session_state.xpts; ypts=st.session_state.ypts
    x=st.session_state.x; npoly=st.session_state.order
    orderlabel = ''
    if (npoly == 1):
        orderlabel='first order fit'
    elif (npoly == 2):
        orderlabel='second order fit'
    elif (npoly == 3):
        orderlabel='third order fit'
    elif (npoly == 4):
        orderlabel='fourth order fit'
    elif (npoly == 5):
        orderlabel='fifth order fit'
    elif (npoly == 6):
        orderlabel='sixth order fit'
        
    # Get the current axis
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('Pox')
    titlestr='Comparison between experimental data points for a '+orderlabel
    plt.title(titlestr)
    plt.plot(xpts,ypts,marker='o',c='r')
    plt.plot(x,fitting(xpts,ypts,x,npoly))
    plt.xlim([0,2])
    plt.ylim([1,4])
    plt.legend(['experimental data',orderlabel])    
    
    return plt

####################
# Msin Streamlit commands
####################
# Streamlit commands
st.title("Polynomial Curve Fitting")
st.write("This application enables you to explore the effect of the order of interpolation")
# read full ddataset
data = read_data()
orders = [1,2,3,4,5,6]

selected_order = st.selectbox('Select Order of Interpolation',orders)

if 'order' not in st.session_state:
    st.session_state.order = 1
    if st.button("Plot Graph and Calculations"):
        figplt = update_curve_plot(data)
        st.pyplot(figplt)
else:
    st.session_state.order = selected_order
    figplt = update_curve_plot(data)
    st.pyplot(figplt)






