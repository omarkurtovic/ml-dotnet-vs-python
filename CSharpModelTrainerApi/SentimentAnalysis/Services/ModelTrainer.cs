using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using SharedCL.SentimentAnalysis.Dtos;
using SharedCL.SentimentAnalysis.Enums;
using SharedCL.SentimentAnalysis.Mappings;
using SharedCL.SentimentAnalysis.Models;
using System.Data;

namespace CSharpModelTrainerApi.SentimentAnalysis.Services
{
    public class ModelTrainer
    {
        public ModelPerformance TrainModel(TrainData trainData)
        {
            MLContext mlContext = new();
            var repoRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));

            var dataPath = Path.Combine(repoRoot, "data", "sentiment-analysis", "IMDB Dataset.csv");
            IDataView data = mlContext.Data.LoadFromTextFile<SentimentData>(dataPath, hasHeader: true, separatorChar: ',', allowQuoting: true);

            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);

            var trainSetDV = mlContext.Data.LoadFromEnumerable(
                mlContext.Data.CreateEnumerable<SentimentData>(split.TrainSet, reuseRowObject: false));
            var testSetDV = mlContext.Data.LoadFromEnumerable(
                mlContext.Data.CreateEnumerable<SentimentData>(split.TestSet, reuseRowObject: false));

            var sentimentCleaner = new SentimentCleanerMapping();
            var featurize = mlContext.Transforms
                .CustomMapping(sentimentCleaner.GetMapping(), contractName: "SentimentCleaner")
                .Append(mlContext.Transforms.Text.FeaturizeText(
                    outputColumnName: "Features",
                    inputColumnName: nameof(SentimentData.Review)));

            IEstimator<ITransformer> trainer = trainData.Algorithm switch
            {
                TrainerAlgorithm.SdcaLogisticRegression  => mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: "SentimentValue", featureColumnName: "Features"),
                TrainerAlgorithm.LbfgsLogisticRegression => mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                    labelColumnName: "SentimentValue", featureColumnName: "Features"),
                TrainerAlgorithm.AveragedPerceptron      => mlContext.BinaryClassification.Trainers.AveragedPerceptron(
                    labelColumnName: "SentimentValue", featureColumnName: "Features"),
                TrainerAlgorithm.LinearSvm               => mlContext.BinaryClassification.Trainers.LinearSvm(
                    labelColumnName: "SentimentValue", featureColumnName: "Features"),
                TrainerAlgorithm.FastTree                => mlContext.BinaryClassification.Trainers.FastTree(
                    labelColumnName: "SentimentValue", featureColumnName: "Features"),
                TrainerAlgorithm.FastForest              => mlContext.BinaryClassification.Trainers.FastForest(
                    labelColumnName: "SentimentValue", featureColumnName: "Features"),
                _ => throw new ArgumentOutOfRangeException(nameof(trainData.Algorithm))
            };

            var pipeline = featurize.Append(trainer);

            var model = pipeline.Fit(trainSetDV);

            var modelDir = Path.Combine(repoRoot, "models", "sentiment-analysis", "csharp");
            Directory.CreateDirectory(modelDir);
            mlContext.Model.Save(model, trainSetDV.Schema, Path.Combine(modelDir, $"{trainData.ModelName}.zip"));

            var trainMetrics = mlContext.BinaryClassification.Evaluate(model.Transform(trainSetDV), labelColumnName: "SentimentValue");
            var testMetrics  = mlContext.BinaryClassification.Evaluate(model.Transform(testSetDV),  labelColumnName: "SentimentValue");

            return new ModelPerformance
            {
                TrainingAccuracy             = trainMetrics.Accuracy,
                TrainingF1Score              = trainMetrics.F1Score,
                TrainingAreaUnderRocCurve    = trainMetrics.AreaUnderRocCurve,
                TrainingPositivePrecision    = trainMetrics.PositivePrecision,
                TrainingPositiveRecall       = trainMetrics.PositiveRecall,
                TestingAccuracy              = testMetrics.Accuracy,
                TestingF1Score               = testMetrics.F1Score,
                TestingAreaUnderRocCurve     = testMetrics.AreaUnderRocCurve,
                TestingPositivePrecision     = testMetrics.PositivePrecision,
                TestingPositiveRecall        = testMetrics.PositiveRecall
            };
        }
    }
}

