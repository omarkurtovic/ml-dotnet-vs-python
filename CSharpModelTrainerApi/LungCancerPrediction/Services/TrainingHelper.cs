using TorchSharp;
using static TorchSharp.torch;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class TrainingHelper
    {
        public static torch.Device GetOptimalDevice()
        {
            return torch.device("cpu");
        }
    }
}
