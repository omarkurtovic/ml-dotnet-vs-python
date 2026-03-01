using CSharpModelTrainer.SharedKernel;
using Microsoft.ML;
using SharedCL.CarValuePrediction.Mappings;
using SharedCL.CarValuePrediction.Models;
using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Text;

namespace CSharpModelTrainer.CarValuePrediction.Services
{
    public class CarValueModelTrainer : IModelTrainer
    {
        public void TrainModel()
        {
            Console.WriteLine("=== Car Price Prediction Model Trainer === ");
            Console.WriteLine("=== Language: C# ===");
            Console.WriteLine();

            MLContext mlContext = new();
            var repoRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));

            // load data
            Console.WriteLine("Loading data...");
            var dataPath = Path.Combine(repoRoot, "data", "car-prediction", "train.csv");
            IDataView data = mlContext.Data.LoadFromTextFile<CarInfoInput>(dataPath, hasHeader: true, separatorChar: ',');
            var allRows = mlContext.Data.CreateEnumerable<CarInfoInput>(data, reuseRowObject: false).ToList();
            Console.WriteLine("Sample data:");
            foreach (var item in allRows.Take(5))
            {
                Console.WriteLine(item);
            }
            Console.WriteLine($"Number of rows: {allRows.Count}");

            // filter data
            Console.WriteLine("Filtering data...");
            Console.WriteLine($"Rows before filtering: {allRows.Count}");
            IDataView filteredData = mlContext.Data.FilterRowsByColumn(data, "Price", lowerBound: 100, upperBound: 200000);
            var filteredCount = mlContext.Data.CreateEnumerable<CarInfoInput>(filteredData, reuseRowObject: false).Count();
            Console.WriteLine($"Rows after filtering: {filteredCount}");


            // split data
            var split = mlContext.Data.TrainTestSplit(filteredData, testFraction: 0.2, seed: 1);
            var trainSet = mlContext.Data
                .CreateEnumerable<CarInfoInput>(split.TrainSet, reuseRowObject: false);

            var testSet = mlContext.Data
                .CreateEnumerable<CarInfoInput>(split.TestSet, reuseRowObject: false);

            var trainSetDV = mlContext.Data.LoadFromEnumerable<CarInfoInput>(trainSet);
            var testSetDV = mlContext.Data.LoadFromEnumerable<CarInfoInput>(testSet);

            // data cleanup
            var mileageCleaner = new MileageCleanerMapping();
            var pipeline = mlContext.Transforms.CustomMapping(mileageCleaner.GetMapping(), contractName: "MileageCleaner")
                .Append(mlContext.Transforms.DropColumns("MileageKm"));


            // one hot encoding
            var multiColumnKeyPipeline =
                    mlContext.Transforms.Categorical.OneHotEncoding(
                        [
                            new InputOutputColumnPair("Manufacturer"),
                            new InputOutputColumnPair("Model"),
                            new InputOutputColumnPair("Category"),
                            new InputOutputColumnPair("LeatherInterior"),
                            new InputOutputColumnPair("FuelType"),
                            new InputOutputColumnPair("EngineVolume"),
                            new InputOutputColumnPair("GearBoxType"),
                            new InputOutputColumnPair("DriveWheels"),
                            new InputOutputColumnPair("Doors"),
                            new InputOutputColumnPair("Wheel"),
                            new InputOutputColumnPair("Color")
                        ]);

            var fullPipeline = pipeline
                .Append(multiColumnKeyPipeline)
                .Append(mlContext.Transforms.Concatenate("Features", [
                    "Manufacturer",
                    "Model",
                    "ProdYear",
                    "Category",
                    "LeatherInterior",
                    "FuelType",
                    "EngineVolume",
                    "Mileage",
                    "Cylinders",
                    "GearBoxType",
                    "DriveWheels",
                    "Doors",
                    "Wheel",
                    "Color",
                    "Airbags"
                ]))
                .Append(mlContext.Regression.Trainers.FastForest(
                    labelColumnName: "Price",
                    featureColumnName: "Features",
                    numberOfTrees: 200,
                    numberOfLeaves: 4096,
                    minimumExampleCountPerLeaf: 1
                ));

            Console.WriteLine("Training model...");
            var model = fullPipeline.Fit(trainSetDV);

            var modelDir = Path.Combine(repoRoot, "models", "car-prediction", "csharp");
            if (!Directory.Exists(modelDir))
            {
                Directory.CreateDirectory(modelDir);
            }

            var modelPath = Path.Combine(modelDir, "csharp_rf_carprice.zip");
            mlContext.Model.Save(model, trainSetDV.Schema, modelPath);

            // === Model Evaluation ===
            var trainPredictions = model.Transform(trainSetDV);
            var testPredictions = model.Transform(testSetDV);

            var trainMetrics = mlContext.Regression.Evaluate(trainPredictions, labelColumnName: "Price");
            var testMetrics = mlContext.Regression.Evaluate(testPredictions, labelColumnName: "Price");

            Console.WriteLine("┌─── TRAINING SET METRICS ───┐");
            Console.WriteLine($"  R²:   {trainMetrics.RSquared:F4}");
            Console.WriteLine($"  RMSE: {trainMetrics.RootMeanSquaredError:F2}");
            Console.WriteLine($"  MAE:  {trainMetrics.MeanAbsoluteError:F2}");
            Console.WriteLine("└─────────────────────────────┘");
            Console.WriteLine();
            Console.WriteLine("┌─── TEST SET METRICS ───┐");
            Console.WriteLine($"  R²:   {testMetrics.RSquared:F4}");
            Console.WriteLine($"  RMSE: {testMetrics.RootMeanSquaredError:F2}");
            Console.WriteLine($"  MAE:  {testMetrics.MeanAbsoluteError:F2}");
            Console.WriteLine("└─────────────────────────┘");
        }
    }
}
