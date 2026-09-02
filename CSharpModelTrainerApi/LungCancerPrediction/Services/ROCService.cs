using CSharpModelTrainerApi.LungCancerPrediction.Models;
using SharedCL;
using System.Runtime.CompilerServices;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class ROCService
    {
        public  List<LCRocDto> CalculateROC(List<LCPredictions> predictions)
        {
            var thresholds = predictions.Select(p => p.MalignantProbability).Distinct().OrderByDescending(x => x).ToList();
            var rocDtos = new List<LCRocDto>
            {
                new LCRocDto() { FalsePositiveRate = 0, TruePositiveRate = 0 }
            };
            foreach (var threshold in thresholds)
            {
                int[,] confusionMatrix = CalculateConfusionMatrix(predictions, threshold);

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

        public double CalculateAUC(List<LCRocDto> rocData)
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

        private int[,] CalculateConfusionMatrix(List<LCPredictions> predictions, double threshold)
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
