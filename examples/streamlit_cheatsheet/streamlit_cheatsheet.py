# streamlit_cheatsheet.py

import streamlit as st
import numpy as np
import time

st.text('Hello world')
number = st.slider("Pick a number",0,100)
file = st.file_uploader("Pick a file")
date = st.date_input("Pick a date")
st.markdown('_Markdown_')
st.caption('Balloons. Hundreds of them ...')
st .latex(r''' e^{i\pi}+1=0''')
st.write('Most objects')
st.title('My title')
st.header('My header')
st.subheader('My sub')
st.code('for i in range(8): foo()')
st.json({'foo':'bar','fu':'ba'})
st.metric(label="Temp", value="273 K", delta="1.2 K")

st.image('./leaveoneout1.png')
col1,col2 = st.columns(2)
col1.write('Column 1')
col2.write('Column 2')
col1, col2, col3 = st.columns([3,1,1])
with col1:
    st.write('This is column 1')

tab1, tab2 = st.tabs(["Tab1", "Tab 2"])
tab1.write("this is tab 1")
tab2.write("this is tab 2")

with tab1:
    st.radio('Select one:', [1,2])
    
# st.stop() 

with st.form(key='my_form'):
    username = st.text_input('Username')
    password = st.text_input('Password')
    st.form_submit_button('Login')

# personalize app for users
#if st.user.email == 'jane@email.com':
#    display_jane_content()
#elif st.user.email == 'adam@foocopr.io':
#    display_adam_content()
#else:
#    st.write('Please contact us to get access')

# Display interactive widgets
st.button('Hit me')
# st.data_editor('Edit data')  # data needs to be a pandas dataframe
st.checkbox('Check me out')
st.radio('Pick one:', ['nose','ear'])
st.selectbox('Select',[1,2,3])
st.multiselect('Multiselect',[1,2,3])
st.slider('Slide me',min_value=0, max_value=10)
st.select_slider('Slide to select',options=[1,'2'])
my_text = st.text_input('Enter some text')
print(my_text)
st.number_input('Enter a number')
st.text_area('Area for textual entry')
st.date_input('Date input')
st.time_input('Time entry')
data = 'hello world'
st.download_button('One the dl',data)
st.camera_input("---")
st.color_picker('Pick a color')

# Use widgets' returned values in variables
for i in range(int(st.number_input('Num:'))): print('Hello')

my_slider_val = st.slider('Quinn Mallory',1,88)
st.write(my_slider_val**2)

with st.chat_message("user"):
    st.write("Hello ")
    st.line_chart(np.random.randn(30,3))

# dsiplay a chat input widget
st.chat_input("Say something")

# display code
st.echo()
with st.echo():
    st.write('Code will be executed and printed')

# display progress and status
with st.spinner(text='In progress'):
    time.sleep(10)
    st.success('Done')

bar = st.progress(50)
time.sleep(3)
bar.progress(100)
