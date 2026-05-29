using CSharpModelTrainerApi.LungCancerPrediction.Datasets;
using CSharpModelTrainerApi.LungCancerPrediction.NeuralNetworks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.TensorExtensionMethods;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using static TorchSharp.torch.utils;
using static TorchSharp.torch.utils.data;
using System.Diagnostics;
using SharedCL;
using CSharpModelTrainerApi.Services;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class LCTrainer(PathResolver pathResolver)
    {
        public  Result<LCDto> TrainModel(LCTrainingParamsDto trainInfo, HardwareInfoService hardwareInfoService)
        {
            if(string.IsNullOrEmpty(trainInfo.Name))
            {
                return Result<LCDto>.Failure("Naziv modela ne smije biti prazan");
            }
            if(trainInfo.Language != ModelLanguageDto.CSharp)
            {
                return Result<LCDto>.Failure("Jezik modela mora biti C#");
            }
            if (trainInfo.Epochs < 1 || trainInfo.Epochs > 10)
            {
                return Result<LCDto>.Failure("Broj epoha mora biti između 1 i 10");
            }

            var modelDB = new LCDto
            {
                Name = trainInfo.Name,
                Language = (ModelLanguageDto)trainInfo.Language,
                EpochData = [],
                HardwareInfo = hardwareInfoService.GetCpuInfo()
            };

            Device defaultDevice = TrainingHelper.GetOptimalDevice();
            torch.set_default_device(defaultDevice);
            var dataDirectory = pathResolver.GetLungCancerDataPath();

            // 1. DATA LOADING BENCHMARK
            var dataLoadingStopwatch = Stopwatch.StartNew();
            var trainingData = new LungCancerTrainDataset(trainInfo.WithFlips, dataDirectory); 
            var classWeights = torch.tensor(trainingData.GetClassWeights()).to(defaultDevice);
            var testData = new LungCancerTestDataset(dataDirectory);
            var trainLoader = torch.utils.data.DataLoader(trainingData, batchSize: 8, shuffle: true, device: defaultDevice);
            var testLoader = torch.utils.data.DataLoader(testData, batchSize: 8, shuffle: false, device: defaultDevice);
            dataLoadingStopwatch.Stop();

            var model = new LungCancerNN().to(defaultDevice);
            var loss = nn.CrossEntropyLoss(classWeights);
            var optimizer = torch.optim.Adam(model.parameters(), lr: 1e-4);
            var epochs = trainInfo.Epochs;

            double trainingTime = 0;
            double validationTime = 0;


            foreach (var epoch in Enumerable.Range(0, epochs))
            {
                // 2. TRAINING BENCHMARK
                Stopwatch trainingStopwatch = Stopwatch.StartNew();
                var trainEpochData = Train(trainLoader, model, loss, optimizer);
                trainingStopwatch.Stop();

                // 2. VALIDATION BENCHMARK
                Stopwatch validationStopwatch = Stopwatch.StartNew();
                var validationEpochData = Validate(testLoader, model, loss);
                validationStopwatch.Stop();

                trainingTime += trainingStopwatch.Elapsed.TotalSeconds;
                validationTime += validationStopwatch.Elapsed.TotalSeconds;

                var epochData = new LCEpochDataDto()
                {
                    Epoch = epoch,
                    TrainingLoss = trainEpochData.Loss,
                    TrainingAccuracy = trainEpochData.Accuracy,
                    ValidationLoss = validationEpochData.Loss,
                    ValidationAccuracy = validationEpochData.Accuracy,
                    BenignPrecision = validationEpochData.BenignPrecision,
                    BenignRecall = validationEpochData.BenignRecall,
                    BenignF1Score = validationEpochData.BenignF1Score,
                    NormalPrecision = validationEpochData.NormalPrecision,
                    NormalRecall = validationEpochData.NormalRecall,
                    NormalF1Score = validationEpochData.NormalF1Score,
                    MalignantPrecision = validationEpochData.MalignantPrecision,
                    MalignantRecall = validationEpochData.MalignantRecall,
                    MalignantF1Score = validationEpochData.MalignantF1Score,
                    MacroPrecision = validationEpochData.MacroPrecision,
                    MacroRecall = validationEpochData.MacroRecall,
                    MacroF1Score = validationEpochData.MacroF1Score,
                    WeightedPrecision = validationEpochData.WeightedPrecision,
                    WeightedRecall = validationEpochData.WeightedRecall,
                    WeightedF1Score = validationEpochData.WeightedF1Score,
                };

                modelDB.EpochData.Add(epochData);
            }

            modelDB.TrainingTimeInSeconds = trainingTime;
            modelDB.ValidationTimeInSeconds = validationTime;
            modelDB.DataLoadingTimeInSeconds = dataLoadingStopwatch.Elapsed.TotalSeconds;
            var modelPath = pathResolver.GetModelPath(trainInfo);
            model.save(modelPath);

            return Result<LCDto>.Success(modelDB);
        }


        private static EpochData Train(DataLoader dataloader, LungCancerNN model, CrossEntropyLoss loss_fn, Adam optimizer)
        {
            var size = dataloader.dataset.Count;
            model.train();

            float totalLoss = 0f;
            int batchCount = 0;
            int[,] confusionMatrix = new int[3, 3];
            long total = dataloader.dataset.Count;

            foreach (var item in dataloader)
            {
                var images = item["image"];
                var correctIndicies = item["label"];

                var predictions = model.call(images);
                var loss = loss_fn.call(predictions, correctIndicies);
                totalLoss += loss.item<float>();
                batchCount++;

                loss.backward();
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm: 1.0);
                optimizer.step();
                optimizer.zero_grad();

                using (var n = torch.no_grad())
                {
                    var winningIndices = predictions.argmax(1);
                    long[] predArray = [.. winningIndices.cpu().data<long>()];
                    long[] labelArray = [.. correctIndicies.cpu().data<long>()];

                    for (int i = 0; i < predArray.Length; ++i)
                    {
                        confusionMatrix[labelArray[i], predArray[i]]++;
                    }
                }

                if (batchCount % 100 == 0)
                {
                    var current = batchCount * images.shape[0];
                    Console.WriteLine($"loss: {loss.item<float>(),7}  [{current,5}/{size,5}]");
                }
            }

            float averageTrainLoss = totalLoss / batchCount;
            return ClassificationReport(confusionMatrix, 3, size, averageTrainLoss);
        }

        private static EpochData Validate(DataLoader dataloader, LungCancerNN model, CrossEntropyLoss loss_fn)
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

                    long[] predArray = [.. winningIndices.cpu().data<long>()];
                    long[] labelArray = [.. correctIndicies.cpu().data<long>()];

                    for(int i = 0; i < predArray.Length; ++i)
                    {
                        confusionMatrix[labelArray[i], predArray[i]]++;
                    }
                }
            }

            float averageValidationLoss = totalLoss / batchCount;
            return ClassificationReport(confusionMatrix, 3, total, averageValidationLoss);
        }


        private static EpochData ClassificationReport(int[,] confusionMatrix, int numClasses, long total, float averageLoss)
        {
            EpochData result = new();
            result.Loss = averageLoss;

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
                    int val = confusionMatrix[i, j];
                    classSupport += val;
                    if (i == j)
                    {
                        truePositives += val;
                        correct += val;
                    }
                    else
                    {
                        falseNegatives += val;
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

            result.Accuracy = correct / (float)total;


            result.MacroPrecision = macroPrecision / numClasses;
            result.MacroRecall = macroRecall / numClasses;
            result.MacroF1Score = macroF1 / numClasses;

            result.WeightedPrecision = weightedPrecision / totalSupport;
            result.WeightedRecall = weightedRecall / totalSupport;
            result.WeightedF1Score = weightedF1 / totalSupport;

            return result;
        }

        public class EpochData
        {
            public float Accuracy { get; set; }
            public float Loss { get; set; }
            public float BenignPrecision { get; set; }
            public float BenignRecall { get; set; }
            public float BenignF1Score { get; set; }
            public float NormalPrecision { get; set; }
            public float NormalRecall { get; set; }
            public float NormalF1Score { get; set; }
            public float MalignantPrecision { get; set; }
            public float MalignantRecall { get; set; }
            public float MalignantF1Score { get; set; }
            public float MacroPrecision { get; set; }
            public float MacroRecall { get; set; }
            public float MacroF1Score { get; set; }
            public float WeightedPrecision { get; set; }
            public float WeightedRecall { get; set; }
            public float WeightedF1Score { get; set; }
        }
    }
}

