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
            var modelDB = new LungCancerModel();
            modelDB.Name = trainInfo.ModelName;
            modelDB.Language = trainInfo.ModelLanguage;
            modelDB.EpochData = new List<LungCancerModelEpochData>();


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
            var optimizer = torch.optim.Adam(model.parameters(), lr: 1e-4);

            var epochs = trainInfo.Epochs;


            foreach (var epoch in Enumerable.Range(0, epochs))
            {
                var trainingLoss = Train(trainLoader, model, loss, optimizer);
                var epochData = Test(testLoader, model, loss);
                epochData.TrainingLoss = trainingLoss;

                modelDB.EpochData.Add(epochData);
            }

            var modelPath = pathResolver.GetModelPath(trainInfo);
            model.save(modelPath);

            return Result<LungCancerModel>.Success(modelDB);
        }


        private static float Train(DataLoader dataloader, LungCancerNN model, CrossEntropyLoss loss_fn, Adam optimizer)
        {
            var size = dataloader.dataset.Count;
            model.train();
            Tensor? loss = null;
            int batch = 0;
            foreach (var item in dataloader)
            {
                var x = item["image"];
                var y = item["label"];

                var pred = model.call(x);

                loss = loss_fn.call(pred, y);

                loss.backward();
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm: 1.0);
                optimizer.step();
                optimizer.zero_grad();

                if (batch % 100 == 0)
                {
                    var current = (batch + 1) * x.shape[0];
                    Console.WriteLine($"loss: {loss.item<float>(),7}  [{current,5}/{size,5}]");
                }

                batch++;
            }

            return loss!.item<float>();
        }
        private static LungCancerModelEpochData Test(DataLoader dataloader, LungCancerNN model, CrossEntropyLoss loss_fn)
        {
            model.eval();

            float totalLoss = 0f;
            int batchCount = 0;
            int[,] confusionMatrix = new int[3, 3];
            long total = dataloader.dataset.Count;

            using (var n = torch.no_grad())
            {
                foreach (var item in dataloader)
                {
                    var images = item["image"];
                    var correctIndicies = item["label"];

                    var predictions = model.call(images); 
                    var loss = loss_fn.call(predictions, correctIndicies);
                    totalLoss += loss.item<float>();
                    batchCount++;

                    var winningIndices = predictions.argmax(1);

                    long[] predArray = winningIndices.cpu().data<long>().ToArray();
                    long[] labelArray = correctIndicies.cpu().data<long>().ToArray();

                    for(int i = 0; i < predArray.Length; ++i)
                    {
                        confusionMatrix[labelArray[i], predArray[i]]++;
                    }
                }
            }

            float averageValidationLoss = totalLoss / batchCount;
            return ClassificationReport(confusionMatrix, 3, total, averageValidationLoss);
        }


        private static LungCancerModelEpochData ClassificationReport(int[,] confusionMatrix, int numClasses, long total, float averageValidationLoss)
        {
            LungCancerModelEpochData result = new();
            result.ValidationLoss = averageValidationLoss;

            float macroPrecision = 0f, macroRecall = 0f, macroF1 = 0f;
            float weightedPrecision = 0f, weightedRecall = 0f, weightedF1 = 0f;
            int totalSupport = 0;
            int correct = 0;

            for (int i = 0; i < numClasses; ++i)
            {
                int truePositives = 0;
                int falsePositives = 0;
                int falseNegatives = 0;

                int classSupport = 0;

                for (int j = 0; j < numClasses; ++j)
                {
                    classSupport += confusionMatrix[i, j];
                    if (i == j)
                    {
                        truePositives += confusionMatrix[i, j];
                        correct += confusionMatrix[i, j];
                    }
                    else
                    {
                        falseNegatives += confusionMatrix[i, j];
                        falsePositives += confusionMatrix[j, i];
                    }
                }

                float precision = (truePositives + falsePositives) == 0 ? 0 :
                  truePositives / (float)(truePositives + falsePositives);

                float recall = (truePositives + falseNegatives) == 0 ? 0 :
                               truePositives / (float)(truePositives + falseNegatives);

                float F1 = (precision + recall) == 0 ? 0 :
                           2 * (precision * recall) / (precision + recall);

                macroPrecision += precision;
                macroRecall += recall;
                macroF1 += F1;

                weightedPrecision += precision * classSupport;
                weightedRecall += recall * classSupport;
                weightedF1 += F1 * classSupport;
                totalSupport += classSupport;


                if (i == 0)
                {
                    result.BenignPrecision = precision;
                    result.BenignRecall = recall;
                    result.BenignF1Score = F1;
                }
                else if(i == 1)
                {
                    result.MalignantPrecision = precision;
                    result.MalignantRecall = recall;
                    result.MalignantF1Score = F1;
                }
                else if(i == 2)
                {
                    result.NormalPrecision = precision;
                    result.NormalRecall = recall;
                    result.NormalF1Score = F1;
                }
            }

            result.ValidationAccuracy = correct / (float)total;


            result.MacroPrecision = macroPrecision / numClasses;
            result.MacroRecall = macroRecall / numClasses;
            result.MacroF1Score = macroF1 / numClasses;

            result.WeightedPrecision = weightedPrecision / totalSupport;
            result.WeightedRecall = weightedRecall / totalSupport;
            result.WeightedF1Score = weightedF1 / totalSupport;

            return result;
        }
    }
}

