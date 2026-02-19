using WebApp.CarValuePrediction.Models;
using WebApp.SentimentAnalysis.Models;

namespace WebApp.SentimentAnalysis.Services
{
    public class SentimentAnalysisPredictionService
    {
        public List<SentimentAnalysisPredictionModel> GetModels()
        {
            var result = new List<SentimentAnalysisPredictionModel>
            {
                new()
                {
                    Name = "C#",
                    Model = SentimentAnalysisPredictionModelType.CSharp
                },
                new()
                {
                    Name = "Python",
                    Model = SentimentAnalysisPredictionModelType.Python
                }
            };

            return result;
        }
    }
}
