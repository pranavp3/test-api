import numpy as np
import pandas as pd
from flask import Flask
import requests
from flask import request
import math
import tensorflow as tf
from flask import jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import sklearn
import json
import shutil
import os
import cv2
import mysql 
from init_2 import ETo, GDD, KC, IMG


# restricting gpu usage
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


app = Flask(__name__)






#classifier =  tf.keras.models.load_model('D:/office/GramworkX_ Image/pest detection/data/data/production/efficientnet_b0_v6_epoch40.h5')
#classifier = tf.keras.models.load_model('/home/ubuntu/flask_app_project/flask_app/efficientnet_b0_v6_epoch40.h5')
classifier = tf.keras.models.load_model(r'C:/Users/prana/Downloads/efficientnet_b0_v6_epoch40.h5')



@app.route('/')
def home():
    return "You are not authorized"

@app.route("/pestdetection/", methods=['POST'])
def pestpredict():
    try:
        if request.method == 'POST':
            ApiKey = request.form.get('ApiKey')
            imagepath = request.form.get('imagepath')
            croptype = request.form.get('croptype')
        else:
            return jsonify(boolresponse=str(False), prediction=str(0), probability=str(0), id=str(0), response="Request type is not POST")
    except:
        return jsonify(boolresponse=str(False), prediction=str(0), probability=str(0), id=str(0), response="Request type is not POST")
    if croptype not in IMG.cropdict.keys():
        return jsonify(boolresponse=str(False), prediction=str(0), probability=str(0), id=str(0), response="Check your crop type")
    else:
        # connect
        mydb = mysql.connector.connect(
            host="ls-25680f515e471efdf54796f570a95d35d48c083c.crx199snke8e.ap-south-1.rds.amazonaws.com",
            user="dbmasteruser",
            password="24i{<H%>>zNnp7{!e<j%U[^[`%uq+j*X",
            database="dbmaster"
        )
        mycursor = mydb.cursor()

        sql1 = "SELECT count(*) FROM api_auth WHERE api_status= 'active' AND api_key = %s"
        key = (ApiKey,)
        mycursor.execute(sql1, key)
        myresult = mycursor.fetchall()

        # close cursor and connection
        mycursor.close()
        mydb.close()
        if myresult[0][0] > 0:
            try:
                r = requests.get(imagepath, stream=True)
                if r.status_code == 200:
                    # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
                    r.raw.decode_content = True
            except Exception as e:
                return jsonify(boolresponse=str(False), prediction=str(0), probability=str(0), id=str(0), response="Problem in fetching image")
            try:
                with open(r'C:/Users/prana/Downloads/sample.jpg', 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            except Exception as e:
                return jsonify(boolresponse=str(False), prediction=str(0), probability=str(0), id=str(0), response="File corrupted")
            try:
                img = load_img(r'C:/Users/prana/Downloads/sample.jpg')
            except:
                return jsonify(boolresponse=str(False), prediction=str(0), probability=str(0), id=str(0), response="Issues with image")            
            try:   # Check for blurriness of the image
                threshold=150.0
                main_img = cv2.imread(r'C:/Users/prana/Downloads/sample.jpg')
                resized_image = cv2.resize(main_img, (224, 224))
                gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                fm = IMG.variance_of_laplacian(gray)
                if float(fm) <= threshold:
                    return jsonify(boolresponse=str(False),prediction = str(0), probability  = str(0),id= str(0), response="Blurry Image, please give a good quality image")
            except:
                pass
            try:
                crop_name= None
                crop_name = IMG.web_entities_crop(r'C:/Users/prana/Downloads/sample.jpg') 
                if crop_name!=None:
                    try:                                                                                                           
                        for entity in crop_name.web_detection.web_entities:
                            if (entity.description == 'leaf' or entity.description == 'Leaf' or entity.description in IMG.cropdict_gcp.keys() or entity.description in IMG.cropdict.keys() or entity.description in IMG.diseasedict.keys() or entity.description in IMG.cropdict_extra.keys()):
                                break
                        if (entity.description != 'leaf' and entity.description != 'Leaf' and entity.description not in IMG.cropdict_gcp.keys() and entity.description not in IMG.cropdict.keys() and entity.description not in IMG.diseasedict.keys() and entity.description not in IMG.cropdict_extra.keys() ) :
                            print(entity)             
                            return jsonify(boolresponse=str(False),prediction = str(0), probability  = str(0),id= str(0), response="Looks like this image is of %s. Please take a leaf image" % (crop_name.web_detection.web_entities[0].description))
                        else:
                            pass                    
                    except:
                        return jsonify(boolresponse=str(False),prediction = str(0), probability  = str(0),id= str(0), response="Looks like this image is not a leaf." )
                    try:                    
                        for entity in crop_name.web_detection.web_entities:
                            if  entity.description in IMG.cropdict.keys() or entity.description in IMG.cropdict_gcp.keys()  or entity.description in IMG.cropdict_extra.keys() :                        
                                break                  
                        leaf = entity.description                      
                        if croptype == leaf:                                                   
                            pass 
                        elif leaf not in IMG.cropdict_gcp.keys() and leaf not in IMG.cropdict.keys() and leaf not in IMG.cropdict_extra.keys() :  # if not in both dictionaries then give it to prediction model
                            pass
                        elif croptype=='Grapes' and leaf in IMG.cropdict_extra.keys():
                            pass               
                        elif leaf  in IMG.cropdict.keys() or leaf in IMG.cropdict_gcp.keys() or leaf in IMG.cropdict_extra.keys():
                            return jsonify(boolresponse=str(False),prediction = str(0), probability  = str(0),id= str(0), response="Looks like this image is of %s leaf. Please send %s leaf image" % (leaf, croptype))               
                        else:
                            pass            
                    except:
                        return jsonify(boolresponse=str(False),prediction = str(0), probability  = str(0),id= str(0), response='Sorry this image cannot be processed')            
                else:
                    return jsonify(boolresponse=str(False),prediction = str(0), probability  = str(0),id= str(0), response='Cannot be processed..')
            except:
                return jsonify(boolresponse=str(False),prediction = str(0), probability  = str(0),id= str(0), response='CV error..')
            
            img = img.resize((224, 224))
            x = img_to_array(img)
            x = x.astype('float32') / 255.
          
            x = x.reshape((1,) + x.shape)
            prob = classifier.predict(x)
            prob_transposed = prob.flatten()
            indices = IMG.cropdict[croptype]

            crop = pd.DataFrame()
            crop['prob'] = prob_transposed[indices]
            crop['croptype'] = croptype
            crop['key'] = indices

            maxvalue = crop['prob'].max()
            maxindex = crop['prob'].idxmax()
            key_at_maxindex = crop.iloc[maxindex]['key']
            prediction = "" + IMG.diseases[key_at_maxindex]
            disease_id = IMG.id_dict[prediction]
            return jsonify(boolresponse=str(True), prediction=str(prediction), probability=str(maxvalue), id=str(disease_id), response="Succeeded")

        else:
            return jsonify(boolresponse=str(False), prediction=str(0), probability=str(0), id=str(0), response="Please Check your API Key")


if __name__ == '__main__':
    app.run()
