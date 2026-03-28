using CSharpModelTrainerApi.LungCancerPrediction.Datasets;
using CSharpModelTrainerApi.LungCancerPrediction.NeuralNetworks;
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
            Device defaultDevice = TrainingHelper.GetOptimalDevice();
            torch.set_default_device(defaultDevice);

            var dataDirectory = pathResolver.GetLungCancerDataPath();

            var trainingData = new LungCancerTrainDataset(dataDirectory); 
            var classWeights = torch.tensor(trainingData.GetClassWeights()).to(defaultDevice);

            var testData = new LungCancerTestDataset(dataDirectory);

            var trainLoader = torch.utils.data.DataLoader(trainingData, batchSize: 8, shuffle: true, device: defaultDevice);
            var testLoader = torch.utils.data.DataLoader(testData, batchSize: 8, shuffle: false, device: defaultDevice);

            var model = new LungCancerNN().to(defaultDevice);
            var loss = nn.CrossEntropyLoss(classWeights);
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
        

        static void Train(DataLoader dataloader, LungCancerNN model, CrossEntropyLoss loss_fn, Adam optimizer)
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
        static (double accuracy, double avgLoss) Test(DataLoader dataloader, LungCancerNN model, CrossEntropyLoss loss_fn)
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
    }
}

