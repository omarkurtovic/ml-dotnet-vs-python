using WebApp.CarValuePrediction.Models;
using WebApp.SharedKernel;

namespace WebApp.SentimentAnalysis.Services
{
    public class SentimentAnalysisPredictionService
    {
        public List<PredictionModel> GetModels()
        {
            var result = new List<PredictionModel>
            {
                new()
                {
                    Name = "C#",
                    Model = PredictionModelType.CSharp
                },
                new()
                {
                    Name = "Python",
                    Model = PredictionModelType.Python
                }
            };

            return result;
        }
    }
}
