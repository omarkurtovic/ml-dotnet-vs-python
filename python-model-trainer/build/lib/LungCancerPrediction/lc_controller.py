
# https://www.kaggle.com/code/adityamahimkar/lung-cancer-prediction-on-image-data/notebook

import numpy as np 
import cv2
import random
from fastapi import APIRouter, HTTPException, UploadFile, File
import time
import asyncio
import threading

import torch
from torch.utils.data import DataLoader

from .models import ModelLanguageDto, LCDto, LCEpochDataDto, LCTrainingParamsDto, EpochData, LCPredictionDto
from .neural_networks import LungCancerNN
from .datasets import LungCancerTrainDataset, LungCancerTestDataset
from .services import HardwareUntils, TrainingHelper, PathResolver, PredictionService


router = APIRouter()

_training_state: dict[int, dict] = {}
_state_lock = threading.Lock()

@router.post("/Python/LungCancer/Predict")
async def predict(model_name: str, file: UploadFile = File(...)) -> LCPredictionDto:
    if model_name == "":
        raise HTTPException(status_code=400, detail="Naziv modela ne smije biti prazan")

    return await PredictionService.predict(model_name, file)


@router.post("/Python/LungCancer/Train")
async def start_training(model_id: int, train_data: LCTrainingParamsDto):
    if train_data.name == "" :
        raise HTTPException(status_code=400, detail="Naziv modela ne smije biti prazan")
    
    if train_data.language != ModelLanguageDto.Python:
        raise HTTPException(status_code=400, detail="Jezik modela mora biti Python")

    if train_data.epochs < 1 or train_data.epochs > 100:
        raise HTTPException(status_code=400, detail="Broj epoha mora biti između 1 i 100")

    with _state_lock:
        _training_state[model_id] = {
            "totalEpochs": train_data.epochs,
            "modelStatusDto": 0,
            "name": train_data.name,
            "language": ModelLanguageDto.Python,
            "trainingTimeInSeconds": 0.0,
            "hardwareInfo": HardwareUntils.get_optimal_hardware_info(),
            "currentEpoch": 0,
        }

    asyncio.create_task(asyncio.to_thread(_run_training, model_id, train_data))
    return {"modelId": model_id}


@router.get("/Python/LungCancer/Train/{model_id}")
def get_training_info(model_id: int):
    with _state_lock:
        state = _training_state.get(model_id)
        return dict(state) if state is not None else None


def _run_training(model_id: int, train_data: LCTrainingParamsDto):
  try:
    model_db = LCDto(
        name=train_data.name,
        language=ModelLanguageDto.Python,
        hardwareInfo=HardwareUntils.get_optimal_hardware_info(),
        trainingTimeInSeconds=0,
        epochData=[])

    default_device = TrainingHelper.get_optimal_device()
    torch.set_default_device(default_device)

    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    data_directory = PathResolver.get_lung_cancer_data_path()
    
    device_generator = torch.Generator(device=default_device)
    training_data = LungCancerTrainDataset(with_flips=train_data.withFlips, data_directory=data_directory)
    class_weights = torch.tensor(training_data.get_class_weights(), dtype=torch.float32).to(default_device)
    test_data = LungCancerTestDataset(data_directory=data_directory)
    train_loader = DataLoader(training_data, batch_size=8, shuffle=True, generator=device_generator)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
    
    model = LungCancerNN().to(default_device)
    loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = train_data.epochs

    total_training_time = 0.0

    for epoch in range(epochs):
        training_start = time.perf_counter()
        train_epoch_data = train(train_loader, model, loss, optimizer)
        training_end = time.perf_counter()

        validation_epoch_data = validate(test_loader, model, loss)

        total_training_time += (training_end - training_start)

        epoch_data = LCEpochDataDto()
        epoch_data.epoch = epoch
        epoch_data.trainingLoss = train_epoch_data.loss
        epoch_data.trainingAccuracy = train_epoch_data.accuracy
        epoch_data.validationLoss = validation_epoch_data.loss
        epoch_data.validationAccuracy = validation_epoch_data.accuracy
        epoch_data.benignPrecision = validation_epoch_data.benignPrecision
        epoch_data.benignRecall = validation_epoch_data.benignRecall
        epoch_data.benignF1Score = validation_epoch_data.benignF1Score
        epoch_data.malignantPrecision = validation_epoch_data.malignantPrecision
        epoch_data.malignantRecall = validation_epoch_data.malignantRecall
        epoch_data.malignantF1Score = validation_epoch_data.malignantF1Score
        epoch_data.normalPrecision = validation_epoch_data.normalPrecision
        epoch_data.normalRecall = validation_epoch_data.normalRecall
        epoch_data.normalF1Score = validation_epoch_data.normalF1Score
        epoch_data.macroPrecision = validation_epoch_data.macroPrecision
        epoch_data.macroRecall = validation_epoch_data.macroRecall
        epoch_data.macroF1Score = validation_epoch_data.macroF1Score
        epoch_data.weightedPrecision = validation_epoch_data.weightedPrecision
        epoch_data.weightedRecall = validation_epoch_data.weightedRecall
        epoch_data.weightedF1Score = validation_epoch_data.weightedF1Score

        model_db.epochData.append(epoch_data)
        with _state_lock:
            _training_state[model_id].update({
                "currentEpoch": epoch + 1,
                "trainingTimeInSeconds": total_training_time,
                "trainingAccuracy": epoch_data.trainingAccuracy,
                "trainingLoss": epoch_data.trainingLoss,
                "validationAccuracy": epoch_data.validationAccuracy,
                "validationLoss": epoch_data.validationLoss,
                "benignPrecision": epoch_data.benignPrecision,
                "benignRecall": epoch_data.benignRecall,
                "benignF1Score": epoch_data.benignF1Score,
                "malignantPrecision": epoch_data.malignantPrecision,
                "malignantRecall": epoch_data.malignantRecall,
                "malignantF1Score": epoch_data.malignantF1Score,
                "normalPrecision": epoch_data.normalPrecision,
                "normalRecall": epoch_data.normalRecall,
                "normalF1Score": epoch_data.normalF1Score,
                "macroPrecision": epoch_data.macroPrecision,
                "macroRecall": epoch_data.macroRecall,
                "macroF1Score": epoch_data.macroF1Score,
                "weightedPrecision": epoch_data.weightedPrecision,
                "weightedRecall": epoch_data.weightedRecall,
                "weightedF1Score": epoch_data.weightedF1Score,
            })

    model_db.trainingTimeInSeconds = total_training_time

    model_path = PathResolver.get_model_path(train_data)
    torch.save(model.state_dict(), model_path)

    with _state_lock:
        _training_state[model_id].update({
            "modelStatusDto": 1,
            "trainingTimeInSeconds": total_training_time,
        })
  except Exception:
    with _state_lock:
        if model_id in _training_state:
            _training_state[model_id]["modelStatusDto"] = 2
    raise



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    total_loss = 0.0
    confusion_matrix = torch.zeros((3, 3), dtype=torch.int32)

    default_device = TrainingHelper.get_optimal_device()

    for batch_count, item in enumerate(dataloader, start=1):
        images = item["image"].to(default_device)
        correct_indices = item["label"].to(default_device)

        predictions = model(images)
        loss = loss_fn(predictions, correct_indices)
        
        total_loss += loss.item()

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            winning_indices = predictions.argmax(dim=1)
            
            for true_label, pred_label in zip(correct_indices, winning_indices):
                confusion_matrix[true_label.item(), pred_label.item()] += 1

        if batch_count % 100 == 0:
            current = batch_count * len(images)
            print(f"loss: {loss.item():>7.5f}  [{current:>5d}/{size:>5d}]")

    average_train_loss = total_loss / batch_count
    return classification_report(confusion_matrix, 3, size, average_train_loss)


def validate(dataloader, model, loss_fn):
    model.eval()

    total_loss = 0.0
    batch_count = 0
    
    confusion_matrix = torch.zeros((3, 3), dtype=torch.int32)
    total = len(dataloader.dataset)
    
    default_device = TrainingHelper.get_optimal_device()

    with torch.no_grad():
        for item in dataloader:
            images = item["image"].to(default_device)
            correct_indices = item["label"].to(default_device)

            predictions = model(images)
            loss = loss_fn(predictions, correct_indices)
            
            total_loss += loss.item()
            batch_count += 1

            winning_indices = predictions.argmax(dim=1)

            for true_label, pred_label in zip(correct_indices, winning_indices):
                confusion_matrix[true_label.item(), pred_label.item()] += 1

    average_validation_loss = total_loss / batch_count
    return classification_report(confusion_matrix, 3, total, average_validation_loss)


def classification_report(confusion_matrix, num_classes, total, average_loss):

    result = EpochData();
    result.loss = average_loss

    macro_precision, macro_recall, macro_f1 = 0.0, 0.0, 0.0
    weighted_precision, weighted_recall, weighted_f1 = 0.0, 0.0, 0.0
    total_support = 0
    correct = 0

    for i in range(num_classes):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        class_support = 0

        for j in range(num_classes):
            val = confusion_matrix[i, j].item()
            class_support += val
            
            if i == j:
                true_positives += val
                correct += val
            else:
                false_negatives += val
                false_positives += confusion_matrix[j, i].item()

        precision = 0.0 if (true_positives + false_positives) == 0 else true_positives / (true_positives + false_positives)
        recall = 0.0 if (true_positives + false_negatives) == 0 else true_positives / (true_positives + false_negatives)
        f1 = 0.0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        weighted_precision += precision * class_support
        weighted_recall += recall * class_support
        weighted_f1 += f1 * class_support
        total_support += class_support

        if i == 0:
            result.benignPrecision = precision
            result.benignRecall = recall
            result.benignF1Score = f1
        elif i == 1:
            result.malignantPrecision = precision
            result.malignantRecall = recall
            result.malignantF1Score = f1
        elif i == 2:
            result.normalPrecision = precision
            result.normalRecall = recall
            result.normalF1Score = f1
    
    result.accuracy = correct / total if total > 0 else 0.0
    result.macroPrecision = macro_precision / num_classes
    result.macroRecall = macro_recall / num_classes
    result.macroF1Score = macro_f1 / num_classes
    result.weightedPrecision = weighted_precision / total_support if total_support > 0 else 0.0
    result.weightedRecall = weighted_recall / total_support if total_support > 0 else 0.0
    result.weightedF1Score = weighted_f1 / total_support if total_support > 0 else 0.0

    return result

