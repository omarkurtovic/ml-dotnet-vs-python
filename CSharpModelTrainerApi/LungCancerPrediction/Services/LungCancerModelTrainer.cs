using CSharpModelTrainerApi.Shared;
using Microsoft.ML;
using SharedCL.LungCancerPrediction.Models;
using SharedCL.SentimentAnalysis.Enums;
using SharedCL.SentimentAnalysis.Mappings;
using SharedCL.Shared.Enums;
using SharedCL.Shared.Models;
using SkiaSharp;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.TensorExtensionMethods;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using static TorchSharp.torch.utils;
using static TorchSharp.torch.utils.data;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class LungCancerModelTrainer(PathResolver pathResolver)
    {
        public  Result<LungCancerModel> TrainModel(LungCancerTrainingParams trainInfo)
        {
            Device defaultDevice = GetOptimalDevice();
            torch.set_default_device(defaultDevice);

            var dataDirectory = pathResolver.GetLungCancerDataPath();

            var trainingData = new LungCancerTrainDataset(dataDirectory);
            var testData = new LungCancerTestDataset(dataDirectory);

            var trainLoader = torch.utils.data.DataLoader(trainingData, batchSize: 64, shuffle: true, device: defaultDevice);
            var testLoader = torch.utils.data.DataLoader(testData, batchSize: 64, shuffle: false, device: defaultDevice);

            var model = new NeuralNetwork().to(defaultDevice);
            var loss = nn.CrossEntropyLoss();
            var optimizer = torch.optim.Adam(model.parameters());

            var epochs = 5;

            foreach (var epoch in Enumerable.Range(0, epochs))
            {
                Console.WriteLine($"Epoch {epoch + 1}\n-------------------------------");
                Train(trainLoader, model, loss, optimizer);
                Test(testLoader, model, loss);
            }

            Console.WriteLine("Done!");

            var modelPath = pathResolver.GetModelPath(trainInfo);
            model.save(modelPath);

            var (valAccuracy, valLoss) = Test(testLoader, model, loss);

            return Result<LungCancerModel>.Success(new LungCancerModel
            {
                Name = trainInfo.ModelName,
                Language = ModelLanguage.CSharp,
                ValidationAccuracy = valAccuracy,
                ValidationLoss = valLoss,
            });
        }

        static void Train(DataLoader dataloader, NeuralNetwork model, CrossEntropyLoss loss_fn, Adam optimizer)
        {
            var size = dataloader.dataset.Count;
            model.train();

            int batch = 0;
            foreach (var item in dataloader)
            {
                var x = item["image"];
                var y = item["label"];

                var pred = model.call(x);

                var loss = loss_fn.call(pred, y);

                loss.backward();

                optimizer.step();

                optimizer.zero_grad();

                if (batch % 100 == 0)
                {
                    loss = loss.item<float>();

                    var current = (batch + 1) * x.shape[0];

                    Console.WriteLine($"loss: {loss.item<float>(),7}  [{current,5}/{size,5}]");
                }

                batch++;
            }
        }
        static (double accuracy, double avgLoss) Test(DataLoader dataloader, NeuralNetwork model, CrossEntropyLoss loss_fn)
        {
            var size = (int)dataloader.dataset.Count;
            var num_batches = (int)dataloader.Count;

            model.eval();

            var test_loss = 0F;
            var correct = 0F;

            using (var n = torch.no_grad())
            {
                foreach (var item in dataloader)
                {
                    var x = item["image"];
                    var y = item["label"];

                    var pred = model.call(x);

                    test_loss += loss_fn.call(pred, y).item<float>();
                    correct += (pred.argmax(1) == y).type(ScalarType.Float32).sum().item<float>();
                }
            }

            test_loss /= num_batches;
            correct /= size;
            Console.WriteLine($"Test Error: \n Accuracy: {(100 * correct):F1}%, Avg loss: {test_loss:F8} \n");

            return (correct, test_loss);
        }



        public class NeuralNetwork : nn.Module<Tensor, Tensor>
        {
            public NeuralNetwork() : base("LungCancerCNN")
            {
                model = Sequential(
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

                RegisterComponents();
            }

            Sequential model;

            public override Tensor forward(Tensor input)
            {
                return model.call(input);
            }
        }

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

    public class LungCancerTrainDataset : Dataset
    {
        private readonly List<string> _imagePaths = [];
        private readonly List<long> _labels = [];

        public LungCancerTrainDataset(string dataDirectory)
        {
            var categories = new List<string> { "Bengin cases", "Malignant cases", "Normal cases" };

            for (int i = 0; i < categories.Count; ++i)
            {
                var path = Path.Join(dataDirectory, categories[i]);

                int categoryImageCount = (int)(Directory.GetFiles(path).Length * 0.75);

                for(int j = 0; j < categoryImageCount; ++j)
                {
                    _imagePaths.Add(Directory.GetFiles(path)[j]);
                    _labels.Add(i);
                }
            }
        }
        public override long Count => _imagePaths.Count;

        public override Dictionary<string, Tensor> GetTensor(long index)
        {
            var path = _imagePaths[(int)index];
            var label = _labels[(int)index];
            
            var imageTensor = ImageLoader.ImageToTensor(path);
            return new Dictionary<string, Tensor>
            {
                ["image"] = imageTensor,
                ["label"] = torch.tensor(label)
            };
        }
    }

    public class LungCancerTestDataset : Dataset
    {
        private readonly List<string> _imagePaths = [];
        private readonly List<long> _labels = [];

        public LungCancerTestDataset(string dataDirectory)
        {
            var categories = new List<string> { "Bengin cases", "Malignant cases", "Normal cases" };

            for (int i = 0; i < categories.Count; ++i)
            {
                var path = Path.Join(dataDirectory, categories[i]);
                var files = Directory.GetFiles(path);
                for (int j = (int)(files.Length * 0.75); j < files.Length; ++j)
                {
                    _imagePaths.Add(files[j]);
                    _labels.Add(i);
                }
            }
        }
        public override long Count => _imagePaths.Count;

        public override Dictionary<string, Tensor> GetTensor(long index)
        {
            var path = _imagePaths[(int)index];
            var label = _labels[(int)index];

            var imageTensor = ImageLoader.ImageToTensor(path);
            return new Dictionary<string, Tensor>
            {
                ["image"] = imageTensor,
                ["label"] = torch.tensor(label)
            };
        }
    }


    public class ImageLoader
    {
        public static Tensor ImageToTensor(string imagePath)
        {
            int imageSize = 256;
            using SKBitmap bitmap = SKBitmap.Decode(imagePath);
            using var resized = bitmap.Resize(new SKImageInfo(imageSize, imageSize, SKColorType.Gray8), SKFilterQuality.Medium);

            float[] bytes = new float[resized.Bytes.Length];
            for(int i = 0; i < resized.Bytes.Length; ++i)
            {
                bytes[i] = resized.Bytes[i] / 255F;
            }
            return torch.tensor(bytes, 1, imageSize, imageSize);
        }
    }
}

