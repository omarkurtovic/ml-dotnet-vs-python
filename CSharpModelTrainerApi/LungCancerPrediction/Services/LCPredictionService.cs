using CSharpModelTrainerApi.LungCancerPrediction.NeuralNetworks;
using CSharpModelTrainerApi.Services;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SharedCL;
using SixLabors.ImageSharp.Processing;
using System.Diagnostics;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using Tensor = TorchSharp.torch.Tensor;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class LCPredictionService(PathResolver pathResolver)
    {
        public async Task<LCPredictionDto> Predict(LCDto dto, IFormFile file)
        {
            if (dto.Language != ModelLanguageDto.CSharp)
                throw new ArgumentException("Invalid model language");

            if (file == null) return null!;

            Device defaultDevice = TrainingHelper.GetOptimalDevice();
            torch.set_default_device(defaultDevice);

            var model = new LungCancerNN().to(defaultDevice);
            var modelPath = pathResolver.GetModelPath(dto);

            model.load(modelPath);
            model.eval();

            Tensor image = await ImageLoader.FormFileImageToTensor(file);
            image = image.to(defaultDevice);

            using (torch.no_grad())
            {
                Stopwatch stopWatch = Stopwatch.StartNew();

                var output = model.call(image);
                var prediction = output.softmax(dim: 1);

                stopWatch.Stop();

                return new LCPredictionDto
                {
                    BenignScore = prediction[0, 0].item<float>(),
                    MalignantScore = prediction[0, 1].item<float>(),
                    NormalScore = prediction[0, 2].item<float>(),
                    PredictionTimeInSeconds = stopWatch.Elapsed.TotalSeconds
                };
            }
        }
    }
}
