
import io
from LungCancerPrediction.models import LCTrainingParamsDto
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from fastapi import UploadFile
from pathlib import Path
import psutil
from cpuinfo import get_cpu_info
from .models import ModelLanguageDto

class ImageLoader:
    IMAGE_SIZE = 256

    @staticmethod
    def image_path_to_tensor(image_path: str) -> torch.Tensor:

        image = Image.open(image_path).convert('L')
        image = image.resize((ImageLoader.IMAGE_SIZE, ImageLoader.IMAGE_SIZE))
        tensor = TF.to_tensor(image)
        return tensor

    @staticmethod
    async def form_file_image_to_tensor(file: UploadFile) -> torch.Tensor:

        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert('L')
        image = image.resize((ImageLoader.IMAGE_SIZE, ImageLoader.IMAGE_SIZE))
        tensor = TF.to_tensor(image)

        tensor = tensor.unsqueeze(0) 
        
        return tensor



class PathResolver:
    @staticmethod
    def get_lung_cancer_data_path() -> Path:
        repo_root = Path("..")
        return repo_root.joinpath('data/lung-cancer-prediction')

    @staticmethod
    def get_model_path(dto: LCTrainingParamsDto) -> Path:
        return PathResolver.get_lc_model_path(dto.name, dto.language)

    @staticmethod
    def get_lc_model_path(model_name: str, language: ModelLanguageDto) -> Path:
        repo_root = Path("..")
        return repo_root.joinpath('models', 'lung-cancer-prediction', 'python',  f'{model_name}.dat')


class TrainingHelper:
    @staticmethod

    def get_optimal_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            return torch.device("mps:0")
        else:
            return torch.device("cpu")


class HardwareUntils:
    @staticmethod
    def get_optimal_hardware_info():

        if torch.cuda.is_available():
            return HardwareUntils._get_gpu_info(0)
        else:
            return HardwareUntils._get_cpu_info()

    @staticmethod
    def _get_cpu_info():

        info = get_cpu_info()
        cpu_name = info.get('brand_raw', 'Nepoznat CPU')
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        return f"{cpu_name} ({physical_cores} Korova / {logical_cores} Threadova)"

    @staticmethod
    def _get_gpu_info(device_index: int = 0):

        gpu_name = torch.cuda.get_device_name(device_index)
        gpu_memory = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
        return f"{gpu_name} (Memorija: {gpu_memory:.2f}GB)"
