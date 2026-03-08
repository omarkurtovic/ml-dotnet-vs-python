using Microsoft.ML;
using SharedCL.SentimentAnalysis.Dtos;
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
            var allRows = mlContext.Data.CreateEnumerable<SentimentData>(data, reuseRowObject: false).ToList();
            
            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);
            var trainSet = mlContext.Data
                .CreateEnumerable<SentimentData>(split.TrainSet, reuseRowObject: false);

            var testSet = mlContext.Data
                .CreateEnumerable<SentimentData>(split.TestSet, reuseRowObject: false);

            var trainSetDV = mlContext.Data.LoadFromEnumerable<SentimentData>(trainSet);
            var testSetDV = mlContext.Data.LoadFromEnumerable<SentimentData>(testSet);

            var sentimentCleaner = new SentimentCleanerMapping();
            var pipeline = mlContext.Transforms.CustomMapping(sentimentCleaner.GetMapping(), contractName: "SentimentCleaner")
                .Append(mlContext.Transforms.Text.FeaturizeText(
                    outputColumnName: "Features",
                    inputColumnName: nameof(SentimentData.Review)))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: "SentimentValue",
                    featureColumnName: "Features"));

            var model = pipeline.Fit(trainSetDV);

            var modelDir = Path.Combine(repoRoot, "models", "sentiment-analysis", "csharp");
            if (!Directory.Exists(modelDir))
            {
                Directory.CreateDirectory(modelDir);
            }

            var modelPath = Path.Combine(modelDir, $"{trainData.ModelName}.zip");
            mlContext.Model.Save(model, trainSetDV.Schema, modelPath);

            var trainPredictions = model.Transform(trainSetDV);
            var testPredictions = model.Transform(testSetDV);

            var trainMetrics = mlContext.BinaryClassification.Evaluate(trainPredictions, labelColumnName: "SentimentValue");
            var testMetrics = mlContext.BinaryClassification.Evaluate(testPredictions, labelColumnName: "SentimentValue");

            var result = new ModelPerformance();
            result.TrainingAccuracy = trainMetrics.Accuracy;
            result.TrainingF1Score = trainMetrics.F1Score;
            result.TrainingAreaUnderRocCurve = trainMetrics.AreaUnderRocCurve;
            result.TrainingPositivePrecision = trainMetrics.PositivePrecision;
            result.TrainingPositiveRecall = trainMetrics.PositiveRecall;

            result.TestingAccuracy = testMetrics.Accuracy;
            result.TestingF1Score = testMetrics.F1Score;
            result.TestingAreaUnderRocCurve = testMetrics.AreaUnderRocCurve;
            result.TestingPositivePrecision = testMetrics.PositivePrecision;
            result.TestingPositiveRecall = testMetrics.PositiveRecall;

            return result;
        }
    }
}
