import streamlit as st
import cv2
import os
import numpy as np



st.title('YoloV5 : détection d\'incendies, de feu')


st.text('La détection autre que par photo est désactivée : tests impossibles sur mon ordinateur sans gpu !')


image_file = st.file_uploader("Upload image", type=['jpeg', 'jpg']) # streamlit function to upload file



if image_file is not None:        

    # décoder et enregistrer l'image
    image_file = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), 1)
    image_file2 = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./yolov5/pred2.jpg', image_file)


    # effectuer la prédiction
    path_abs = os.path.join(os.getcwd(), 'yolov5', 'pred2.jpg')
    st.text(path_abs)
    os.system(f'python yolov5/detect.py --weights best.pt --img 640 --conf 0.25 --source {path_abs}')
    
    
    # récupérer la dernière prédiction
    path_dossier = 'yolov5/runs/detect'
    if len(os.listdir(path_dossier)) != 1:
        dernier_id_dossier = sorted([int(x.replace('exp', '')) for x in os.listdir(path_dossier) if x !='exp'])[-1]
        dernier_dossier = 'exp' + str(dernier_id_dossier)
    else:
        dernier_dossier = 'exp'

    img_path = os.path.join(path_dossier, dernier_dossier, os.listdir(os.path.join(path_dossier, dernier_dossier))[0])


    # affichage
    st.sidebar.text('Image originale :')
    st.sidebar.image(image_file2, width=240)

    st.text('prédiction :')
    st.image(img_path)

        


