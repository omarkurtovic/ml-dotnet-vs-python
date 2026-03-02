using WebApp.SharedKernel;

namespace WebApp.LungCancerPrediction.Services
{
    public class LungCancerPredictionService
    {
        public static List<PredictionModel> GetModels()
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
