using CSharpModelTrainer.SharedKernel;
using Microsoft.ML;
using SharedCL.SentimentAnalysis.Models;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Text;
using TorchSharp;
using static TorchSharp.TensorExtensionMethods;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;



namespace CSharpModelTrainer.LungCancerPrediction.Services
{
    // https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification
    public class LungCancerPredictionModelTrainer : IModelTrainer
    {
        public void TrainModel()
        {
            Console.WriteLine("=== Lung Cancer Prediction Model Trainer ===");
            Console.WriteLine("=== Language: C# ===");
            Console.WriteLine();

            var repoRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));
            var directory = Path.Join(repoRoot, "data", "lung-cancer-prediction");
            var categories = new List<string> { "Bengin cases", "Malignant cases", "Normal cases" };
            int imageSize = 256;

            Console.WriteLine("Loading data...");
            var data = new List<(float[] pixels, int label)>();

            foreach (var category in categories)
            {
                int classNum = categories.IndexOf(category);
                var path = Path.Join(directory, category);

                foreach (var filepath in Directory.GetFiles(path))
                {
                    using var original = SKBitmap.Decode(filepath);
                    using var resized = original.Resize(new SKImageInfo(imageSize, imageSize, SKColorType.Gray8), SKFilterQuality.Medium);

                    var pixelBytes = resized.Bytes;
                    var pixels = new float[imageSize * imageSize];
                    for (int i = 0; i < pixels.Length; i++)
                        pixels[i] = pixelBytes[i] / 255.0f;

                    data.Add((pixels, classNum));
                }
            }


            var rng = new Random(10);
            data = [.. data.OrderBy(_ => rng.Next())];

            int trainSize = (int)(data.Count * 0.75);

            var trainData = data.Take(trainSize).ToList();
            var validData = data.Skip(trainSize).ToList();


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

            var optimizer = torch.optim.Adam(model.parameters());
            int epochs = 5;
            int batchSize = 8;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                model.train();
                float trainLoss = 0;
                int total = 0;
                long correct = 0;
                foreach (var (X, y) in MakeBatches(trainData, batchSize, imageSize, augment: true))
                {
                    optimizer.zero_grad();
                    var output = model.forward(X);

                    var loss = cross_entropy(output, y);
                    loss.backward();
                    optimizer.step();

                    trainLoss += loss.item<float>();
                    correct += output.argmax(1).eq(y).sum().item<long>();
                    total += (int)y.shape[0];
                }

                model.eval();
                long valCorrect = 0, valTotal = 0;

                using (torch.no_grad())
                {
                    foreach (var (X, y) in MakeBatches(validData, batchSize, imageSize, augment: false))
                    {
                        var output = model.forward(X);
                        valCorrect += output.argmax(1).eq(y).sum().item<long>();
                        valTotal += (int)y.shape[0];
                    }
                }

                Console.WriteLine($"Epoch {epoch + 1}/{epochs} " +
                                  $"- loss: {trainLoss / total:F4} " +
                                  $"- accuracy: {(float)correct / total:F4} " +
                                  $"- val_accuracy: {(float)valCorrect / valTotal:F4}");
            }

            model.eval();

            var modelDir = Path.Join(repoRoot, "models", "lung-cancer-prediction", "csharp");
            Directory.CreateDirectory(modelDir);
            var modelPath = Path.Join(modelDir, "lung-cancer-model.weights");

            model.save(modelPath);

            Console.WriteLine($"Model weights saved → {modelPath}");

        }

        private static IEnumerable<(torch.Tensor X, torch.Tensor y)> MakeBatches(
            List<(float[] pixels, int label)> data, int batchSize, int imageSize, bool augment)
        {
            var rng = new Random();

            for (int i = 0; i < data.Count; i += batchSize)
            {
                var batch = data.Skip(i).Take(batchSize).ToList();

                var allPixels = new float[batch.Count * imageSize * imageSize];
                var allLabels = new long[batch.Count];

                for (int j = 0; j < batch.Count; j++)
                {
                    float[] pixels = augment ? RandomFlip(batch[j].pixels, imageSize, rng) : batch[j].pixels;

                    Array.Copy(pixels, 0, allPixels, j * imageSize * imageSize, pixels.Length);

                    allLabels[j] = batch[j].label;
                }

                var X = torch.tensor(allPixels).reshape(batch.Count, 1, imageSize, imageSize);
                var y = torch.tensor(allLabels);

                yield return (X, y);
            }
        }

        private static float[] RandomFlip(float[] pixels, int imageSize, Random rng)
        {
            var result = (float[])pixels.Clone();

            if (rng.NextDouble() > 0.5)
                for (int y = 0; y < imageSize; y++)
                    for (int x = 0; x < imageSize / 2; x++)
                        (result[y * imageSize + x], result[y * imageSize + (imageSize - 1 - x)]) =
                        (result[y * imageSize + (imageSize - 1 - x)], result[y * imageSize + x]);

            if (rng.NextDouble() > 0.5)
                for (int y = 0; y < imageSize / 2; y++)
                    for (int x = 0; x < imageSize; x++)
                        (result[y * imageSize + x], result[(imageSize - 1 - y) * imageSize + x]) =
                        (result[(imageSize - 1 - y) * imageSize + x], result[y * imageSize + x]);

            return result;
        }
    }
}
