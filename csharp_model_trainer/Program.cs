using CSharpModelTrainer.CarValuePrediction.Services;

class Program
{
    static void Main(string[] args)
    {
        while (true)
        {
            Console.WriteLine();
            Console.WriteLine("=== ML.NET Model Trainer ===");
            Console.WriteLine("1. Train Car Price Model (FastForest)");
            Console.WriteLine("0. Exit");
            Console.Write("Select: ");

            var input = Console.ReadLine()?.Trim();

            switch (input)
            {
                case "1":
                    var carTrainer = new CarValueModelTrainer();
                    carTrainer.TrainModel();
                    break;
                case "0":
                    return;
                default:
                    Console.WriteLine("Invalid option.");
                    break;
            }
        }
    }
}