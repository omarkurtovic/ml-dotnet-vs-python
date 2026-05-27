
# https://www.kaggle.com/code/adityamahimkar/lung-cancer-prediction-on-image-data/notebook

import numpy as np 
import cv2
import random
import os
import torch
from pathlib import Path
from fastapi import APIRouter, HTTPException
import time
from .models import ModelLanguageDto, LCDto, LCEpochDataDto, LCTrainingParamsDto
from .hardware_utils import get_optimal_hardware_info
from training_helper import TrainingHelper
from path_resolver import PathResolver

router = APIRouter()

repo_root = Path("..")
directory = repo_root.joinpath('data/lung-cancer-prediction')



@router.post("/Python/LungCancer/Train")
def train(train_data: LCTrainingParamsDto):
    if train_data.name == "" :
        raise HTTPException(status_code=400, detail="Naziv modela ne smije biti prazan")
    
    if train_data.language != ModelLanguageDto.Python:
        raise HTTPException(status_code=400, detail="Jezik modela mora biti Python")

    if train_data.epochs < 1 or train_data.epochs > 10:
        raise HTTPException(status_code=400, detail="Broj epoha mora biti između 1 i 10")

    model_db = LCDto()
    model_db.name = train_data.name
    model_db.language = ModelLanguageDto.Python
    model_db.epochData = []

    default_device = TrainingHelper.get_optimal_device()
    torch.set_default_device(default_device)

    model_db.hardwareInfo = get_optimal_hardware_info()
    
    data_directory = PathResolver.get_lung_cancer_data_path()
    
    training_data = 
    
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
    
    train_datagen = ImageDataGenerator()
    if(train_data.withFlips):
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


    start_time = time.perf_counter()
    history = model.fit(train_generator, epochs=train_data.epochs, validation_data=val_generator, class_weight=new_weights)
    end_time = time.perf_counter()


    dbModel = LCDto(
        name=train_data.name,
        language=ModelLanguageDto.Python,
        epochData=[],
        trainingTimeInSeconds=int(end_time - start_time),
        
    )

    for epoch in range(train_data.epochs):
        epoch_data = LCEpochDataDto(
        epoch=epoch,
        trainingLoss=history.history['loss'][epoch],
        trainingAccuracy=history.history['accuracy'][epoch],
        validationAccuracy=history.history['val_accuracy'][epoch],
        validationLoss=history.history['val_loss'][epoch])
        dbModel.epochData.append(epoch_data)

    y_pred = model.predict(X_valid, verbose=1)

    y_pred_bool = np.argmax(y_pred, axis=1)

    report = classification_report(y_valid, y_pred_bool, output_dict=True)

    dbModel.epochData[-1].benignPrecision = report["0"]['precision']
    dbModel.epochData[-1].benignRecall = report["0"]['recall']
    dbModel.epochData[-1].benignF1Score = report["0"]['f1-score']
    dbModel.epochData[-1].malignantPrecision = report["1"]['precision']
    dbModel.epochData[-1].malignantRecall = report["1"]['recall']
    dbModel.epochData[-1].malignantF1Score = report["1"]['f1-score']
    dbModel.epochData[-1].normalPrecision = report["2"]['precision']
    dbModel.epochData[-1].normalRecall = report["2"]['recall']
    dbModel.epochData[-1].normalF1Score = report["2"]['f1-score']
    dbModel.epochData[-1].macroPrecision = report["macro avg"]['precision']
    dbModel.epochData[-1].macroRecall = report["macro avg"]['recall']
    dbModel.epochData[-1].macroF1Score = report["macro avg"]['f1-score']
    dbModel.epochData[-1].weightedPrecision = report["weighted avg"]['precision']
    dbModel.epochData[-1].weightedRecall = report["weighted avg"]['recall']
    dbModel.epochData[-1].weightedF1Score = report["weighted avg"]['f1-score']  




    # Save Model
    import tensorflow as tf
    import tf2onnx
    import onnx

    model_dir = repo_root / "models" / "lung-cancer-prediction" / "python"
    model_dir.mkdir(parents=True, exist_ok=True)

    input_signature = [tf.TensorSpec([None, img_size, img_size, 1], tf.float32, name='x')]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    onnx.save(onnx_model, model_dir / f"{train_data.name}.onnx")

    return dbModel

