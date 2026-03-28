using TorchSharp;
using static TorchSharp.torch;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class TrainingHelper
    {
        public static torch.Device GetOptimalDevice()
        {
            Device defaultDevice = default!;
            if (torch.cuda.is_available())
            {
                defaultDevice = torch.device("cuda", index: 0);
            }
            else if (torch.mps_is_available())
            {
                defaultDevice = torch.device("mps", index: 0);
            }
            else
            {
                defaultDevice = torch.device("cpu");
            }

            return defaultDevice;
        }
    }
}
