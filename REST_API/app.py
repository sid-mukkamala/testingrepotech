from fastapi import FastAPI, UploadFile, File
import uvicorn
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
# from PIL import Image    


model = keras.models.load_model('models/bee')

# Config values to be stored in a config file
dim_r, dim_c = (300,150)

# List of outputs in order of their representation in final layer
outputs = ['cooling', 'pollen', 'varroa', 'wasps']



app = FastAPI()

@app.get("/")
def read_root():
    return {"API status": "Bee cool!"}



# 1/1 [==============================] - 1s 1s/step
# {'prediction': 'cooling', 'prediction_probability': 0.9980092}
# 1/1 [==============================] - 0s 56ms/step
# {'prediction': 'pollen', 'prediction_probability': 0.9740272}
# 1/1 [==============================] - 0s 56ms/step
# {'prediction': 'varroa', 'prediction_probability': 1.0}
# 1/1 [==============================] - 0s 54ms/step
# {'prediction': 'wasps', 'prediction_probability': 0.9993197}



@app.post("/predict/")
def predict_price(image_file: bytes = File()):

    # Check if uploaded file is of the correct type
    #  I will check for a few common image types 

    # if image_file.content_type not in ['image/jpeg', 'image/png']:
    #     raise HTTPException(400,detail='Wrong image type')
    
    img = BytesIO(image_file)
    
    img = image.load_img(img, target_size=(dim_r, dim_c))
    
    img_array = image.img_to_array(img)

    img_batch = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_batch)
    
    # Once we have the prediction, we need to identify the ouput class/category it belongs to
    index = np.argmax(prediction[0])

    predicted_class = outputs[index]
    prediction_probability = prediction[0][index]

    return {"prediction": predicted_class, "prediction_probability": float(prediction_probability)}
    









# @app.post("/predict/")
# def predict_price(image_file_path: str):

#     # Check if uploaded file is of the correct type
#     #  I will check for a few common image types 

#     # if image_file.content_type not in ['image/jpeg', 'image/png']:
#     #     raise HTTPException(400,detail='Wrong image type')
    
#     # we can now preprocess the image before giving it to model.predict
#     img = image.load_img(image_file_path, target_size=(dim_r, dim_c))

#     img_array = image.img_to_array(img)
    
#     img_batch = np.expand_dims(img_array, axis=0)
    
#     prediction = model.predict(img_batch)
    
#     # Once we have the prediction, we need to identify the ouput class/category it belongs to
#     index = np.argmax(prediction[0])

#     predicted_class = outputs[index]
#     prediction_probability = prediction[0][index]

#     # We can collect the predicted information and store it in a DB
#     # useful in times of development

#     return {"prediction": predicted_class, "prediction_probability":prediction_probability }