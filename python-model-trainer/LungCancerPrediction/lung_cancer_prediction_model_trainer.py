
# https://www.kaggle.com/code/adityamahimkar/lung-cancer-prediction-on-image-data/notebook

import numpy as np 
import matplotlib.pyplot as plt
import cv2
import random
import os
import imageio
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from pathlib import Path
from fastapi import APIRouter, HTTPException
from enum import IntEnum
from pydantic import BaseModel

router = APIRouter()

repo_root = Path("..")
directory = repo_root.joinpath('data/lung-cancer-prediction')

class ModelLanguage(IntEnum):
    CSharp = 0
    Python = 1


class TrainData(BaseModel):
    modelName: str
    modelLanguage: ModelLanguage



@router.post("/Python/LungCancer/Train")
def train(train_data: TrainData):
    
    categories = ['Bengin cases', 'Malignant cases', 'Normal cases']
    # just showing the different image sizes in the dataset
    # size_data = {}
    # for i in categories:
    #     path = os.path.join(directory, i)
    #     class_num = categories.index(i)
    #     temp_dict = {}
    #     for file in os.listdir(path):
    #         filepath = os.path.join(path, file)
    #         height, width, channels = imageio.imread(filepath).shape
    #         if str(height) + ' x ' + str(width) in temp_dict:
    #             temp_dict[str(height) + ' x ' + str(width)] += 1 
    #         else:
    #             temp_dict[str(height) + ' x ' + str(width)] = 1
    
    #     size_data[i] = temp_dict
        
    # print(size_data)

    # showing a sample image from each category
    # for i in categories:
    #     path = os.path.join(directory, i)
    #     class_num = categories.index(i)
    #     for file in os.listdir(path):
    #         filepath = os.path.join(path, file)
    #         print(i)
    #         img = cv2.imread(filepath, 0)
    #         plt.imshow(img)
    #         plt.show()
    #         break


    # showing some images from all categories after resizing and blurring
    # img_size = 256
    # for i in categories:
    #     cnt, samples = 0, 3
    #     fig, ax = plt.subplots(samples, 3, figsize=(15, 15))
    #     fig.suptitle(i)
    
    #     path = os.path.join(directory, i)
    #     class_num = categories.index(i)
    #     for curr_cnt, file in enumerate(os.listdir(path)):
    #         filepath = os.path.join(path, file)
    #         img = cv2.imread(filepath, 0)
        
    #         img0 = cv2.resize(img, (img_size, img_size))
        
    #         img1 = cv2.GaussianBlur(img0, (5, 5), 0)
        
    #         ax[cnt, 0].imshow(img)
    #         ax[cnt, 1].imshow(img0)
    #         ax[cnt, 2].imshow(img1)
    #         cnt += 1
    #         if cnt == samples:
    #             break
        
    # plt.show()



    data = []
    img_size = 256

    for i in categories:
        path = os.path.join(directory, i)
        class_num = categories.index(i)
        for file in os.listdir(path):
            filepath = os.path.join(path, file)
            img = cv2.imread(filepath, 0)
            # preprocess here
            img = cv2.resize(img, (img_size, img_size))
            data.append([img, class_num])
        
    random.shuffle(data)

    X, y = [], []
    for feature, label in data:
        X.append(feature)
        y.append(label)
    
    print('X length:', len(X))
    print('y counts:', Counter(y))

    # normalize
    X = np.array(X).reshape(-1, img_size, img_size, 1)
    X = X / 255.0
    y = np.array(y)



    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=10, stratify=y)

    print(len(X_train), X_train.shape)
    print(len(X_valid), X_valid.shape)



    new_weights = {
        0: X_train.shape[0]/(3*Counter(y_train)[0]),
        1: X_train.shape[0]/(3*Counter(y_train)[1]),
        2: X_train.shape[0]/(3*Counter(y_train)[2]),
    }


    train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True) 
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(X_train, y_train, batch_size=8) 
    val_generator = val_datagen.flow(X_valid, y_valid, batch_size=8)


    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(16))
    model.add(Dense(3, activation='softmax'))
    model.output_names=['output']
    model.summary()


    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    history = model.fit(train_generator, epochs=5, validation_data=val_generator, class_weight=new_weights)


    y_pred = model.predict(X_valid, verbose=1)

    y_pred_bool = np.argmax(y_pred, axis=1)

    report = classification_report(y_valid, y_pred_bool, output_dict=True)

    print(confusion_matrix(y_true=y_valid, y_pred=y_pred_bool))

    # Save Model
    import tensorflow as tf
    import tf2onnx
    import onnx

    model_dir = repo_root / "models" / "lung-cancer-prediction" / "python"
    model_dir.mkdir(parents=True, exist_ok=True)

    input_signature = [tf.TensorSpec([None, img_size, img_size, 1], tf.float32, name='x')]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    onnx.save(onnx_model, model_dir / f"{train_data.modelName}.onnx")

    return {
        "name":               train_data.modelName,
        "language":           ModelLanguage.Python,
        "trainingAccuracy":   history.history['accuracy'][-1],
        "validationAccuracy": history.history['val_accuracy'][-1],
        "validationLoss":     history.history['val_loss'][-1],
        "benignPrecision":    report["0"]['precision'],
        "benignRecall":       report["0"]['recall'],
        "benignF1Score":      report["0"]['f1-score'],
        "malignantPrecision": report["1"]['precision'],
        "malignantRecall":    report["1"]['recall'],
        "malignantF1Score":   report["1"]['f1-score'],
        "normalPrecision":    report["2"]['precision'],
        "normalRecall":       report["2"]['recall'],
        "normalF1Score":      report["2"]['f1-score'],
        "macroPrecision":     report["macro avg"]['precision'],
        "macroRecall":        report["macro avg"]['recall'],
        "macroF1Score":       report["macro avg"]['f1-score'],
        "weightedPrecision":  report["weighted avg"]['precision'],
        "weightedRecall":     report["weighted avg"]['recall'],
        "weightedF1Score":    report["weighted avg"]['f1-score']
    }

