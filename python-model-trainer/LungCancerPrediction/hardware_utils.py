import tensorflow as tf
from cpuinfo import get_cpu_info
import psutil

def _get_cpu_info():
    info = get_cpu_info()
    cpu_name = info.get('brand_raw', 'Nepoznat CPU')
    pysical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    return f"{cpu_name} ({pysical_cores} Korova / {logical_cores} Threadova)"

def _get_gpu_info(gpu_device):
    details = tf.config.experimental.get_device_details(gpu_device)
    gpu_name = details.get('device_name', 'Nepoznat GPU')
    gpu_memory = details.get('memory_limit', 0) / (1024 ** 3)
    return f"{gpu_name} (Memorija: {gpu_memory}GB)"

def get_optimal_hardware_info():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return _get_gpu_info(gpus[0])
    else:
        return _get_cpu_info()