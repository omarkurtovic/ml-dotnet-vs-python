namespace WebApp.CarValuePrediction.Models
{
    public class CarPricePredictionModel
    {
        public string Name { get; set; } = null!;
        public CarPricePredictionModelType Model { get; set; }
    }
    public enum CarPricePredictionModelType
    {
        CSharpFastForest,
        PythonRandomForestRegressor
    }
}
