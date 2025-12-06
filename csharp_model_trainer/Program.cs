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
        MLContext mlContext = new();

        // load data
        IDataView data = mlContext.Data.LoadFromTextFile<CarInfo>("../data/train.csv", hasHeader: true, separatorChar: ',');

        // filter data
        IDataView filteredData = mlContext.Data.FilterRowsByColumn(data, "Price", lowerBound: 100, upperBound: 200000);

        // split
        var split = mlContext.Data.TrainTestSplit(filteredData, testFraction: 0.2);
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
                .Append(mlContext.Regression.Trainers.FastTree(
                    labelColumnName: "Price",
                    featureColumnName: "Features",
                    numberOfTrees: 200,
                    numberOfLeaves: 30,
                    minimumExampleCountPerLeaf: 10,
                    learningRate: 0.1
                ));

            var model = fullPipeline.Fit(trainSetDV);

            mlContext.Model.Save(model, trainSetDV.Schema, "../models/csharp/csharp_rf_carprice.zip");

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