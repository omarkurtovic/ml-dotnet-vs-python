using WebApp.CarValuePrediction.Models;

namespace WebApp.CarValuePrediction.Services
{
    public class CarPricePredictionService
    {
        public List<CarPricePredictionModel> GetModels()
        {
            var result = new List<CarPricePredictionModel>
            {
                new()
                {
                    Name = "C# FastForest",
                    Model = CarPricePredictionModelType.CSharpFastForest
                },
                new()
                {
                    Name = "Python RandomForestRegressor",
                    Model = CarPricePredictionModelType.PythonRandomForestRegressor
                }
            };

            return result;
        }
    }

    
}
