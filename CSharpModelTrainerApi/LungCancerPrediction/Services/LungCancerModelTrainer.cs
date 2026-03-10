using Microsoft.ML;
using SharedCL.LungCancerPrediction.Models;
using SharedCL.SentimentAnalysis.Enums;
using SharedCL.SentimentAnalysis.Mappings;
using SharedCL.Shared.Enums;
using SharedCL.Shared.Models;
using SkiaSharp;
using TorchSharp;
using static TorchSharp.TensorExtensionMethods;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class LungCancerModelTrainer
    {
        public Result<LungCancerModel> TrainModel(LungCancerTrainingParams trainInfo)
        {
            var repoRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));
            var directory = Path.Join(repoRoot, "data", "lung-cancer-prediction");
            var categories = new List<string> { "Bengin cases", "Malignant cases", "Normal cases" };
            int imageSize = 256;

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
            var modelPath = Path.Join(modelDir, $"{trainInfo.ModelName}.weights");
            model.save(modelPath);



            model.eval();
            var allPreds = new List<int>();
            var allLabels = new List<int>();

            using (torch.no_grad())
            {
                foreach (var (X, y) in MakeBatches(validData, batchSize, imageSize, augment: false))
                {
                    var output = model.forward(X);
                    var preds = output.argmax(1);
                    allPreds.AddRange(preds.data<long>().Select(p => (int)p));
                    allLabels.AddRange(y.data<long>().Select(l => (int)l));
                }
            }

            var categoryNames = new[] { "Benign", "Malignant", "Normal" };
            int numClasses = categoryNames.Length;

            var tp = new int[numClasses];
            var fp = new int[numClasses];
            var fn = new int[numClasses];

            for (int i = 0; i < allLabels.Count; i++)
            {
                int pred = allPreds[i];
                int label = allLabels[i];
                if (pred == label) tp[pred]++;
                else { fp[pred]++; fn[label]++; }
            }

            double Precision(int c) => (tp[c] + fp[c]) == 0 ? 0 : (double)tp[c] / (tp[c] + fp[c]);
            double Recall(int c) => (tp[c] + fn[c]) == 0 ? 0 : (double)tp[c] / (tp[c] + fn[c]);
            double F1(int c) => (Precision(c) + Recall(c)) == 0 ? 0 : 2 * Precision(c) * Recall(c) / (Precision(c) + Recall(c));

            double valAccuracy = (double)allPreds.Zip(allLabels).Count(p => p.First == p.Second) / allLabels.Count;

            return Result<LungCancerModel>.Success(new LungCancerModel
            {
                Name = trainInfo.ModelName,
                Language = ModelLanguage.CSharp,
                ValidationAccuracy = valAccuracy,
                BenignPrecision = Precision(0),
                BenignRecall = Recall(0),
                BenignF1Score = F1(0),
                MalignantPrecision = Precision(1),
                MalignantRecall = Recall(1),
                MalignantF1Score = F1(1),
                NormalPrecision = Precision(2),
                NormalRecall = Recall(2),
                NormalF1Score = F1(2),
                MacroPrecision = Enumerable.Range(0, numClasses).Average(Precision),
                MacroRecall = Enumerable.Range(0, numClasses).Average(Recall),
                MacroF1Score = Enumerable.Range(0, numClasses).Average(F1),
            });

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

