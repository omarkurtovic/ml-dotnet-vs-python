using TorchSharp;
using static TorchSharp.torch;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class TrainingHelper
    {
        public static torch.Device GetOptimalDevice()
        {
            if (torch.cuda_is_available())
            {
                return torch.CUDA;
            }
            else
            {
                return torch.CPU;
            }
        }
    }
}
