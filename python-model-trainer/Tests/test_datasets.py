import pytest
import torch

from LungCancerPrediction.datasets import LungCancerTrainDataset 

def test_get_class_weights_zero_labels_returns_zero_weights():

    dataset = LungCancerTrainDataset.from_params(images=[], labels=[], categories=["Class1", "Class2", "Class3"])
    
    weights = dataset.get_class_weights()

    assert len(weights) == 3
    assert weights[0] == 0
    assert weights[1] == 0
    assert weights[2] == 0

def test_get_class_weights_zero_classes_some_labels_returns_zero_weights():

    dataset = LungCancerTrainDataset.from_params(images=[], labels=[0, 0, 1, 1, 1], categories=[])

    weights = dataset.get_class_weights()

    assert len(weights) == 0

def test_get_class_weights_missing_class_avoids_division_by_zero():

    dataset = LungCancerTrainDataset.from_params(images=[], labels=[0, 0, 2, 2], categories=["Class1", "Class2", "Class3"])
    
    weights = dataset.get_class_weights()

    assert len(weights) == 3
    assert weights[0] == pytest.approx(4/6)
    assert weights[1] == 0
    assert weights[2] == pytest.approx(4/6)

def test_get_class_weights_unbalanced_classes_returns_correct_weights():
    dataset = LungCancerTrainDataset.from_params(images=[], labels=[0, 0, 1, 2, 2], categories=["Class1", "Class2", "Class3"])
    
    weights = dataset.get_class_weights()
    assert len(weights) == 3
    assert weights[0] == pytest.approx(5/6)
    assert weights[1] == pytest.approx(5/3)
    assert weights[2] == pytest.approx(5/6)