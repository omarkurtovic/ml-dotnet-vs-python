using CSharpModelTrainerApi.LungCancerPrediction.Datasets;
using CSharpModelTrainerApi.LungCancerPrediction.Models;
using CSharpModelTrainerApi.LungCancerPrediction.NeuralNetworks;
using CSharpModelTrainerApi.Services;
using SharedCL;
using System.Diagnostics;
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
    public class LCTrainer(PathResolver pathResolver, LCRepository lungCancerModelRepository)
    {
        private LCRepository LungCancerModelRepository { get; set; } = lungCancerModelRepository;
        public async Task<Result<LCDto>> TrainModelAsync(int modelId, LCTrainingParamsDto trainInfo)
        {
            var modelResult = await LungCancerModelRepository.GetModel(modelId);
            if (!modelResult.IsSuccess)
            {
                return Result<LCDto>.Failure("Failed to retrieve model.");
            }
            var modelDB = modelResult.Data!;

            Device defaultDevice = TrainingHelper.GetOptimalDevice();
            torch.set_default_device(defaultDevice);

            long seed = 42;
            torch.manual_seed(seed);
            if (torch.cuda.is_available())
            {
                torch.cuda.manual_seed(seed);
                torch.cuda.manual_seed_all(seed);
            }


            var dataDirectory = pathResolver.GetLungCancerDataPath();

            var trainingData = new LungCancerTrainDataset(dataDirectory, withAugmentation: trainInfo.WithAugmentation);
            var classWeights = torch.tensor(trainingData.GetClassWeights()).to(defaultDevice);
            var testData = new LungCancerTestDataset(dataDirectory);
            var trainLoader = torch.utils.data.DataLoader(trainingData, batchSize: 8, shuffle: true, device: defaultDevice);
            var testLoader = torch.utils.data.DataLoader(testData, batchSize: 8, shuffle: false, device: defaultDevice);

            var model = new LungCancerNN().to(defaultDevice);
            var loss = nn.CrossEntropyLoss(classWeights);
            var optimizer = torch.optim.Adam(model.parameters(), lr: 1e-4);
            var epochs = trainInfo.Epochs;

            double trainingTime = 0;

            foreach (var epoch in Enumerable.Range(0, epochs))
            {
                Stopwatch trainingStopwatch = Stopwatch.StartNew();
                var trainEpochData = Train(trainLoader, model, loss, optimizer);
                trainingStopwatch.Stop();

                var validationEpochData = Validate(testLoader, model, loss, epoch == epochs - 1);

                trainingTime += trainingStopwatch.Elapsed.TotalSeconds;

                var epochData = new LCEpochData()
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
                    LCPredictions = [.. validationEpochData.Predictions.Select(p => new LCPredictions
                    {
                        BenignProbability = p.BenignProbability,
                        MalignantProbability = p.MalignantProbability,
                        NormalProbability = p.NormalProbability,
                        TrueLabel = p.TrueLabel
                    })]
                };

                var addEpochResult = await LungCancerModelRepository.AddEpochData(modelDB.Id, epochData);
                if (!addEpochResult.IsSuccess)
                {
                    await LungCancerModelRepository.UpdateStatusAsync(modelId, Enums.ModelStatus.Failed);
                    return Result<LCDto>.Failure("Greška prilikom spremanja podataka epohe");
                }

                await LungCancerModelRepository.UpdateTrainingTimeAsync(modelId, trainingTime);

            }

            modelDB.TrainingTimeInSeconds = trainingTime;
            var modelPath = pathResolver.GetModelPath(trainInfo);
            model.save(modelPath);

            await LungCancerModelRepository.UpdateStatusAsync(modelId, Enums.ModelStatus.Trained);

            return Result<LCDto>.Success(modelDB);
        }


        private static SegmentEpochData Train(DataLoader dataloader, LungCancerNN model, CrossEntropyLoss loss_fn, Adam optimizer)
        {
            var epochData = new SegmentEpochData();
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
            ClassificationReport(ref epochData, confusionMatrix, 3, size, averageTrainLoss);
            return epochData;
        }

        private static SegmentEpochData Validate(DataLoader dataloader, LungCancerNN model, CrossEntropyLoss loss_fn, bool isLastEpoch = false)
        {
            var epochData = new SegmentEpochData();
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

                    long[] labelArray = [.. correctIndicies.cpu().data<long>()];

                    if (isLastEpoch)
                    {
                        var probs = torch.nn.functional.softmax(predictions, dim: 1).cpu();
                        for (int i = 0; i < predictions.shape[0]; ++i)
                        {
                            var benign = probs[i][0].item<float>();
                            var malignant = probs[i][1].item<float>();
                            var normal = probs[i][2].item<float>();
                            epochData.Predictions.Add(new LCEpochPredictionDto
                            {
                                BenignProbability = benign,
                                MalignantProbability = malignant,
                                NormalProbability = normal,
                                TrueLabel = (int)labelArray[i]
                            });
                        }
                    }

                    var winningIndices = predictions.argmax(1);
                    long[] predArray = [.. winningIndices.cpu().data<long>()];

                    for (int i = 0; i < predArray.Length; ++i)
                    {
                        confusionMatrix[labelArray[i], predArray[i]]++;
                    }
                }
            }

            float averageValidationLoss = totalLoss / batchCount;
            ClassificationReport(ref epochData, confusionMatrix, 3, total, averageValidationLoss);
            return epochData;
        }


        private static void ClassificationReport(ref SegmentEpochData epochData, int[,] confusionMatrix, int numClasses, long total, float averageLoss)
        {
            epochData.Loss = averageLoss;

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
                    epochData.BenignPrecision = precision;
                    epochData.BenignRecall = recall;
                    epochData.BenignF1Score = F1;
                }
                else if (i == 1)
                {
                    epochData.MalignantPrecision = precision;
                    epochData.MalignantRecall = recall;
                    epochData.MalignantF1Score = F1;
                }
                else if (i == 2)
                {
                    epochData.NormalPrecision = precision;
                    epochData.NormalRecall = recall;
                    epochData.NormalF1Score = F1;
                }
            }

            epochData.Accuracy = correct / (float)total;


            epochData.MacroPrecision = macroPrecision / numClasses;
            epochData.MacroRecall = macroRecall / numClasses;
            epochData.MacroF1Score = macroF1 / numClasses;

            epochData.WeightedPrecision = weightedPrecision / totalSupport;
            epochData.WeightedRecall = weightedRecall / totalSupport;
            epochData.WeightedF1Score = weightedF1 / totalSupport;
        }

        public class SegmentEpochData
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
            public List<LCEpochPredictionDto> Predictions { get; set; } = [];
        }
    }
}

