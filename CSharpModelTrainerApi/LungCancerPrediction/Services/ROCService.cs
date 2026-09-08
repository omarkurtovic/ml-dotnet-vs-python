using CSharpModelTrainerApi.LungCancerPrediction.Models;
using SharedCL;
using System.Runtime.CompilerServices;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class LCRocResult
    {
        public List<LCRocDto> Points { get; set; } = [];
        public double Auc { get; set; }
    }
    public class ROCService
    {
        public static LCRocResult Calculate(List<LCValidationScore> scores)
        {
            var points = CalculateROC(scores);
            return new LCRocResult { Points = points, Auc = CalculateAUC(points) };
        }

        private static List<LCRocDto> CalculateROC(List<LCValidationScore> validationScores)
        {
            var thresholds = validationScores.Select(p => p.MalignantProbability).Distinct().OrderByDescending(x => x).ToList();
            var rocDtos = new List<LCRocDto>
            {
                new() { FalsePositiveRate = 0, TruePositiveRate = 0 }
            };
            foreach (var threshold in thresholds)
            {
                int[,] confusionMatrix = CalculateConfusionMatrix(validationScores, threshold);

                double tpr = 0, fpr = 0;
                if ((confusionMatrix[0, 0] + confusionMatrix[1, 0]) != 0)
                {
                    tpr = (double)confusionMatrix[0, 0] / (confusionMatrix[0, 0] + confusionMatrix[1, 0]);
                }
                if((confusionMatrix[0, 1] + confusionMatrix[1, 1]) != 0)
                {
                    fpr = (double)confusionMatrix[0, 1] / (confusionMatrix[0, 1] + confusionMatrix[1, 1]);
                }
                rocDtos.Add(new LCRocDto { TruePositiveRate = tpr, FalsePositiveRate = fpr, Threshold = threshold });
            }
            rocDtos.Add(new LCRocDto() { FalsePositiveRate = 1, TruePositiveRate = 1 });

            return rocDtos;
        }

        private static double CalculateAUC(List<LCRocDto> rocData)
        {
            double auc = 0.0;
            for (int i = 1; i < rocData.Count; i++)
            {
                double x1 = rocData[i - 1].FalsePositiveRate;
                double y1 = rocData[i - 1].TruePositiveRate;

                double x2 = rocData[i].FalsePositiveRate;
                double y2 = rocData[i].TruePositiveRate;

                auc += (x2 - x1) * (y1 + y2) / 2.0;
            }
            return auc;
        }

        private static int[,] CalculateConfusionMatrix(List<LCValidationScore> predictions, double threshold)
        {
            int[,] result = new int[2, 2];
            for(int i = 0; i < predictions.Count; i++)
            {
                int trueLabel = predictions[i].TrueLabel;
                int predictedLabel = predictions[i].MalignantProbability >= threshold ? 1 : 0;

                // true positive
                if (predictedLabel == 1 && trueLabel == 1)
                {
                    result[0, 0]++;
                }
                // false positive
                else if (predictedLabel == 1 && trueLabel != 1)
                {
                    result[0, 1]++;
                }
                // false negative
                else if (predictedLabel != 1 && trueLabel == 1)
                {
                    result[1, 0]++;
                }
                // true negative
                else if (predictedLabel != 1 && trueLabel != 1)
                {
                    result[1, 1]++;
                }
            }

            return result;
        }
    }
}
