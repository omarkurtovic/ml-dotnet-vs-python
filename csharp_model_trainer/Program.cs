using Microsoft.ML;
using Microsoft.ML.Data;

class Program
{
    public class CarInfo
    {

        [LoadColumn(1)]
        public float Price { get; set; }

        [LoadColumn(3)]
        public string Manufacturer { get; set; } = null!;

        [LoadColumn(4)]
        public string Model { get; set; } = null!;

        [LoadColumn(5)]
        public float ProdYear { get; set; }

        [LoadColumn(6)]
        public string Category { get; set; } = null!;

        [LoadColumn(7)]
        public string LeatherInterior { get; set; } = null!;

        [LoadColumn(8)]
        public string FuelType { get; set; } = null!;

        [LoadColumn(9)]
        public string EngineVolume { get; set; } = null!;

        [LoadColumn(10)]
        public string MileageKm { get; set; } = null!;

        [LoadColumn(11)]
        public float Cylinders { get; set; }

        [LoadColumn(12)]
        public string GearBoxType { get; set; } = null!;

        [LoadColumn(13)]
        public string DriveWheels { get; set; } = null!;

        [LoadColumn(14)]
        public string Doors { get; set; } = null!;

        [LoadColumn(15)]
        public string Wheel { get; set; } = null!;

        [LoadColumn(16)]
        public string Color { get; set; } = null!;

        [LoadColumn(17)]
        public float Airbags { get; set; }

        public override string ToString()
        {
            return $"Price: {Price}, {Manufacturer} {Model}, Year: {ProdYear}, Cat: {Category}, Leather: {LeatherInterior}, Fuel: {FuelType}, Engine: {EngineVolume}, Mileage: {MileageKm}, Cyl: {Cylinders}, Gear: {GearBoxType}, Drive: {DriveWheels}, Doors: {Doors}, Wheel: {Wheel}, Color: {Color}, Airbags: {Airbags}";
        }
    }

    public class MileageClean
    {
        public float Mileage { get; set; }
    }

    public class CarInfoEncoded
    {
        public float Price { get; set; }

        public float[] Manufacturer { get; set; } = null!;

        public float[] Model { get; set; } = null!;

        public float ProdYear { get; set; }

        public float[] Category { get; set; } = null!;

        public float[] LeatherInterior { get; set; } = null!;

        public float[] FuelType { get; set; } = null!;
        public float[] EngineVolume { get; set; } = null!;
        public float Mileage { get; set; }

        public float Cylinders { get; set; }

        public float[] GearBoxType { get; set; } = null!;

        public float[] DriveWheels { get; set; } = null!;

        public float[] Doors { get; set; } = null!;

        public float[] Wheel { get; set; } = null!;

        public float[] Color { get; set; } = null!;
        public float Airbags { get; set; }
    }


    static void Main(string[] args)
    {
        Console.WriteLine("=== Car Price Prediction Model Trainer === ");
        Console.WriteLine("=== Language: C# ===");
        Console.WriteLine();

        MLContext mlContext = new();
        var repoRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));

        // load data
        Console.WriteLine("Loading data...");
        var dataPath = Path.Combine(repoRoot, "data", "car-prediction", "train.csv"); 
        IDataView data = mlContext.Data.LoadFromTextFile<CarInfo>(dataPath, hasHeader: true, separatorChar: ',');
        var allRows = mlContext.Data.CreateEnumerable<CarInfo>(data, reuseRowObject: false).ToList();
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
        var filteredCount = mlContext.Data.CreateEnumerable<CarInfo>(filteredData, reuseRowObject: false).Count();
        Console.WriteLine($"Rows after filtering: {filteredCount}");


        // split data
        var split = mlContext.Data.TrainTestSplit(filteredData, testFraction: 0.2, seed: 1);
        var trainSet = mlContext.Data
            .CreateEnumerable<CarInfo>(split.TrainSet, reuseRowObject: false);

        var testSet = mlContext.Data
            .CreateEnumerable<CarInfo>(split.TestSet, reuseRowObject: false);

        var trainSetDV = mlContext.Data.LoadFromEnumerable<CarInfo>(trainSet);
        var testSetDV = mlContext.Data.LoadFromEnumerable<CarInfo>(testSet);

        // data cleanup
        static void milageCleanup(CarInfo input, MileageClean output)
        {
            var mileageStr = input.MileageKm?.Replace(" km", "").Trim();
            output.Mileage = float.TryParse(mileageStr, out var mileage) ? mileage : 0f;
        }

        var pipeline = mlContext.Transforms.CustomMapping((Action<CarInfo, MileageClean>)milageCleanup, contractName: "MileageCleaner")
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
            if(!Directory.Exists(modelDir))
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