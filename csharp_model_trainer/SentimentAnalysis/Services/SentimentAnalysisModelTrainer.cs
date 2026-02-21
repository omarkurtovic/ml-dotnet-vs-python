using CSharpModelTrainer.SharedKernel;
using Microsoft.ML;
using SharedCL.SentimentAnalysis.Mappings;
using SharedCL.SentimentAnalysis.Models;
using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace CSharpModelTrainer.SentimentAnalysis.Services
{
    public class SentimentAnalysisModelTrainer : IModelTrainer
    {
        public void TrainModel()
        {
            Console.WriteLine("=== Sentiment Analysis Model Trainer === ");
            Console.WriteLine("=== Language: C# ===");
            Console.WriteLine();

            MLContext mlContext = new();
            var repoRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));

            // load data
            Console.WriteLine("Loading data...");
            var dataPath = Path.Combine(repoRoot, "data", "sentiment-analysis", "IMDB Dataset.csv");
            IDataView data = mlContext.Data.LoadFromTextFile<SentimentData>(dataPath, hasHeader: true, separatorChar: ',', allowQuoting: true);
            var allRows = mlContext.Data.CreateEnumerable<SentimentData>(data, reuseRowObject: false).ToList();
            Console.WriteLine("Sample data:");
            foreach (var item in allRows.Take(5))
            {
                Console.WriteLine(item);
            }
            Console.WriteLine($"Number of rows: {allRows.Count}");


            // split data
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

            Console.WriteLine("Training model...");
            var model = pipeline.Fit(trainSetDV);
            Console.WriteLine("Model trained.");

            var modelDir = Path.Combine(repoRoot, "models", "sentiment-analysis", "csharp");
            if (!Directory.Exists(modelDir))
            {
                Directory.CreateDirectory(modelDir);
            }

            var modelPath = Path.Combine(modelDir, "csharp_rf_sentiment_analysis.zip");
            mlContext.Model.Save(model, trainSetDV.Schema, modelPath);

            // === Model Evaluation ===
            var trainPredictions = model.Transform(trainSetDV);
            var testPredictions = model.Transform(testSetDV);

            var trainMetrics = mlContext.BinaryClassification.Evaluate(trainPredictions, labelColumnName: "SentimentValue");
            var testMetrics = mlContext.BinaryClassification.Evaluate(testPredictions, labelColumnName: "SentimentValue");

            Console.WriteLine("┌─── TRAINING SET METRICS ───┐");
            Console.WriteLine($"  Accuracy:  {trainMetrics.Accuracy:F4}");
            Console.WriteLine($"  F1 Score:  {trainMetrics.F1Score:F4}");
            Console.WriteLine($"  AUC:       {trainMetrics.AreaUnderRocCurve:F4}");
            Console.WriteLine($"  Precision: {trainMetrics.PositivePrecision:F4}");
            Console.WriteLine($"  Recall:    {trainMetrics.PositiveRecall:F4}");
            Console.WriteLine("└─────────────────────────────┘");
            Console.WriteLine();
            Console.WriteLine("┌─── TEST SET METRICS ───┐");
            Console.WriteLine($"  Accuracy:  {testMetrics.Accuracy:F4}");
            Console.WriteLine($"  F1 Score:  {testMetrics.F1Score:F4}");
            Console.WriteLine($"  AUC:       {testMetrics.AreaUnderRocCurve:F4}");
            Console.WriteLine($"  Precision: {testMetrics.PositivePrecision:F4}");
            Console.WriteLine($"  Recall:    {testMetrics.PositiveRecall:F4}");
            Console.WriteLine("└─────────────────────────┘");
        }
    }
}
