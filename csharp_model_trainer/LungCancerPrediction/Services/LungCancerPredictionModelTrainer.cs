using CSharpModelTrainer.SharedKernel;
using Microsoft.ML;
using SharedCL.SentimentAnalysis.Models;
using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;

using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using static TorchSharp.TensorExtensionMethods;



namespace CSharpModelTrainer.LungCancerPrediction.Services
{
    // https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification
    public class LungCancerPredictionModelTrainer : IModelTrainer
    {
        public void TrainModel()
        {
            Console.WriteLine("=== Lung Cancer Prediction Model Trainer === ");
            Console.WriteLine("=== Language: C# ===");
            Console.WriteLine();

            MLContext mlContext = new();
            var repoRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));

            List<string> categories = new List<string> { "Bengin cases", "Malignant cases", "Normal cases"};


            int imageSize = 256;

            var model = Sequential();
            model.add_module("conv2d", Conv2d(in_channels: 1, out_channels: 64, kernel_size: 3 ));
            model.add_module("relu1", ReLU());
            model.add_module("maxpooling2d1", MaxPool2d((2, 2)));

            model.add_module("conv2d2", Conv2d(in_channels: 64, out_channels: 64, kernel_size: 3));
            model.add_module("relu2", ReLU());
            model.add_module("maxpooling2d2", MaxPool2d((2, 2)));

            model.add_module("flatten", Flatten());
            model.add_module("dense1", Linear(Parameter(16)));
            model.add_module("dense2", Linear(Parameter(3)), Softmax());

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


            //Append ImageClassification trainer to the pipeline containing any preprocessing transforms.
            pipeline
                .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(featureColumnName: "Image")
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel");

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Use the model for inferencing.
            var predictedData = model.Transform(newData).GetColumn<string>("PredictedLabel");
        }
    }
}
