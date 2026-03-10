using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SharedCL.LungCancerPrediction;
using TorchSharp;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class LungCancerPredictionService
    {
        public
        private async Task<LungCancerPredictionModel> PredictWithTorchSharp(string repoRoot)
        {
            if (_file == null) return null!;

            const int imgSize = 256;
            var modelPath = Path.Combine(repoRoot, "models", "lung-cancer-prediction", "csharp", "lung-cancer-model.weights");

            var resizedFile = await _file.RequestImageFileAsync("image/png", imgSize, imgSize);
            using var stream = resizedFile.OpenReadStream(maxAllowedSize: 10 * 1024 * 1024);
            using var image = await SixLabors.ImageSharp.Image.LoadAsync<SixLabors.ImageSharp.PixelFormats.L8>(stream);

            var floatData = new float[imgSize * imgSize];
            for (int y = 0; y < imgSize; y++)
                for (int x = 0; x < imgSize; x++)
                    floatData[y * imgSize + x] = image[x, y].PackedValue / 255.0f;

            var model = Sequential(
                ("conv2d", Conv2d(in_channels: 1, out_channels: 64, kernel_size: 3)),
                ("relu1", ReLU()),
                ("maxpooling2d1", MaxPool2d((2, 2))),
                ("conv2d2", Conv2d(in_channels: 64, out_channels: 64, kernel_size: 3)),
                ("relu2", ReLU()),
                ("maxpooling2d2", MaxPool2d((2, 2))),
                ("flatten", Flatten()),
                ("dense1", Linear(inputSize: 246016, outputSize: 16)),
                ("dense2", Linear(inputSize: 16, outputSize: 3))
            );

            model.load(modelPath);
            model.eval();

            using var input = torch.tensor(floatData).reshape(1, 1, imgSize, imgSize);
            using var output = model.forward(input);
            using var probs = softmax(output, dim: 1);
            var scores = probs.data<float>().ToArray();

            return new LungCancerPredictionModel
            {
                BenignScore = scores[0],
                MalignantScore = scores[1],
                NormalScore = scores[2]
            };
        }


        private async Task<LungCancerPredictionModel> PredictWithOnnx(string repoRoot)
        {
            if (_file == null)
                return null!;

            var modelPath = Path.Combine(repoRoot, "models", "lung-cancer-prediction", "python", "lung_cancer_prediction.onnx");

            const int imgSize = 256;

            var resizedFile = await _file.RequestImageFileAsync("image/png", imgSize, imgSize);
            using var stream = resizedFile.OpenReadStream(maxAllowedSize: 10 * 1024 * 1024);
            using var image = await SixLabors.ImageSharp.Image.LoadAsync<SixLabors.ImageSharp.PixelFormats.L8>(stream);

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
