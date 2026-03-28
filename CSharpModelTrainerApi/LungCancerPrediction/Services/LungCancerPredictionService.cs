using CSharpModelTrainerApi.LungCancerPrediction.Datasets;
using CSharpModelTrainerApi.LungCancerPrediction.NeuralNetworks;
using CSharpModelTrainerApi.Shared;
using Microsoft.AspNetCore.Http;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SharedCL.LungCancerPrediction.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using Tensor = TorchSharp.torch.Tensor;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class LungCancerPredictionService(PathResolver pathResolver)
    {
        public async Task<LungCancerPredictionModel> Predict(LungCancerModel model, IFormFile file)
        {
            if (model.Language == SharedCL.Shared.Enums.ModelLanguage.CSharp)
                return await PredictWithTorchSharp(model, file);
            else if (model.Language == SharedCL.Shared.Enums.ModelLanguage.Python)
                return await PredictWithOnnx(model, file);
            else
                throw new ArgumentException("Invalid model language");
        }

        private async Task<LungCancerPredictionModel> PredictWithTorchSharp(LungCancerModel dbModel, IFormFile file)
        {
            if (file == null) return null!;

            Device defaultDevice = TrainingHelper.GetOptimalDevice();
            torch.set_default_device(defaultDevice);

            var model = new LungCancerNN().to(defaultDevice);
            var modelPath = pathResolver.GetModelPath(dbModel);

            model.load(modelPath);
            model.eval();

            Tensor image = await ImageLoader.FormFileImageToTensor(file);
            image = image.to(defaultDevice);
            
            using (torch.no_grad())
            {
                var output = model.call(image);
                var prediction = output.softmax(dim: 1);

                return new LungCancerPredictionModel
                {
                    BenignScore = prediction[0, 0].item<float>(),
                    MalignantScore = prediction[0, 1].item<float>(),
                    NormalScore = prediction[0, 2].item<float>()
                };
            }


            
        }

        private async Task<LungCancerPredictionModel> PredictWithOnnx(LungCancerModel dbModel, IFormFile file)
        {
            if (file == null) return null!;

            var modelPath = pathResolver.GetModelPath(dbModel);
            const int imgSize = 256;

            using var stream = file.OpenReadStream();
            using var image = await SixLabors.ImageSharp.Image.LoadAsync<SixLabors.ImageSharp.PixelFormats.L8>(stream);
            image.Mutate(x => x.Resize(imgSize, imgSize));

            var floatData = new float[imgSize * imgSize];
            for (int y = 0; y < imgSize; y++)
                for (int x = 0; x < imgSize; x++)
                    floatData[y * imgSize + x] = image[x, y].PackedValue / 255.0f;

            using var session = new InferenceSession(modelPath);
            var inputName = session.InputMetadata.Keys.First();
            var tensor = new DenseTensor<float>(floatData, [1, imgSize, imgSize, 1]);
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensor) };

            using var results = session.Run(inputs);
            var scores = results[0].AsEnumerable<float>().ToArray();

            return new LungCancerPredictionModel
            {
                BenignScore = scores[0],
                MalignantScore = scores[1],
                NormalScore = scores[2]
            };
        }
    }
}
