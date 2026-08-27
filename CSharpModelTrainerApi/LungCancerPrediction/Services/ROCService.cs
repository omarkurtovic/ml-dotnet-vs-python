using CSharpModelTrainerApi.LungCancerPrediction.Models;
using SharedCL.LungCancerPrediction.Dtos;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class ROCService
    {
        public  List<LCRocDto> CalculateROC(List<LCPredictions> predictions)
        {
            var thresholds = predictions.Select(p => p.MalignantProbability).Distinct().OrderByDescending(x => x).ToList();
            var rocDtos = new List<LCRocDto>();
            foreach (var threshold in thresholds)
            {
                int[,] confusionMatrix = CalculateConfusionMatrix(predictions, threshold);

                double tpr = (double)confusionMatrix[0, 0] / (confusionMatrix[0, 0] + confusionMatrix[0, 1]);
                double fpr = (double)confusionMatrix[1, 0] / (confusionMatrix[1, 0] + confusionMatrix[1, 1]);
                rocDtos.Add(new LCRocDto { TruePositiveRate = tpr, FalsePositiveRate = fpr });
            }
            return rocDtos;
        }

        private int[,] CalculateConfusionMatrix(List<LCPredictions> predictions, double threshold)
        {
            int[,] result = new int[2, 2];
            for(int i = 0; i < predictions.Count; i++)
            {
                int trueLabel = predictions[i].TrueLabel;
                int predictedLabel = predictions[i].MalignantProbability >= threshold ? 1 : 0;

                if (trueLabel == 1 && predictedLabel == 1)
                {
                    result[0, 0]++;
                }
                else if (trueLabel == 1 && predictedLabel == 0)
                {
                    result[0, 1]++;
                }
                else if (trueLabel == 0 && predictedLabel == 1)
                {
                    result[1, 0]++;
                }
                else if (trueLabel == 0 && predictedLabel == 0)
                {
                    result[1, 1]++;
                }
            }

            return result;
        }
    }
}
