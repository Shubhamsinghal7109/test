import cv2
import streamlit as st
import numpy as np
import tempfile
face_cascade = cv2.CascadeClassifier('facedata.xml')
choice=st.sidebar.selectbox('My Menu',('Home','Image','Video'))
if choice=='Home':
    st.title('Face Detection')
    st.header('WELCOME')
    st.image('http://governbetter.co/wp-content/uploads/2019/10/Facial-Recognition-Indian-Government.gif')
if choice=='Image':
    st.title('Computer Vision')
    data=st.file_uploader('Upload Image')
    if data:
        bytes_data = data.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        faces = face_cascade.detectMultiScale(img, 1.3, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        st.image(img,channels="BGR")
if choice=='Video':
    data= st.file_uploader("Upload file")
    myimage = st.empty()
    if data:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(data.read())
        vid = cv2.VideoCapture(tfile.name)
        while vid.isOpened():
            ret, frame = vid.read()  
            faces = face_cascade.detectMultiScale(frame, 1.3, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            myimage.image(frame,channels='BGR')

