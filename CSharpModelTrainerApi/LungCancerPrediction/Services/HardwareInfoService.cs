using Hardware.Info;
using TorchSharp;
using static TorchSharp.torch;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class HardwareInfoService
    {
        private readonly IHardwareInfo hardwareInfo = new HardwareInfo();

        public string GetHardwareInfo()
        {
            Device defaultDevice = TrainingHelper.GetOptimalDevice();
            if (defaultDevice.type == DeviceType.CPU)
            {
                return GetCpuInfo();
            }
            else
            {
                return GetGpuInfo();
            }
        }
        public string GetCpuInfo()
        {
            hardwareInfo.RefreshCPUList();
            var cpu = hardwareInfo.CpuList.FirstOrDefault();

            string cpuName = cpu?.Name?.Trim() ?? "Nepoznat CPU";
            uint physicalCores = cpu?.NumberOfCores ?? 0;
            uint logicalCores = cpu?.NumberOfLogicalProcessors ?? (uint)System.Environment.ProcessorCount;

            return $"{cpuName} ({physicalCores} Korova / {logicalCores} Threadova)";
        }

        public string GetGpuInfo()
        {
            hardwareInfo.RefreshVideoControllerList();
            var gpu = hardwareInfo.VideoControllerList.FirstOrDefault();

            string gpuName = gpu?.Name?.Trim() ?? "Nepoznat GPU";
            string gpuMemory = gpu != null ? $"{gpu.AdapterRAM / (1024 * 1024 * 1024)} GB" : "";
            return $"{gpuName} (Memorija: {gpuMemory}GB)";
        }
    }
}
