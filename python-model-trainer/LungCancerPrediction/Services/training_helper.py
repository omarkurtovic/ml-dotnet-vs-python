
import torch

class TrainingHelper:
    @staticmethod
    def get_optimal_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            return torch.device("mps:0")
        else:
            return torch.device("cpu")