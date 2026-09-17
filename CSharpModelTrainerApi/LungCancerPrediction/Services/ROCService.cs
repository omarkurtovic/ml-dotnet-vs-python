using CSharpModelTrainerApi.LungCancerPrediction.Models;
using SharedCL;
using System.Runtime.CompilerServices;
using TorchSharp.Modules;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class LCRocResult
    {
        public List<LCRocDto> Points { get; set; } = [];
        public double Auc { get; set; }
    }
    public class ROCService
    {
        public static LCRocResult Calculate(List<LCValidationScore> scores, int classOfInterest)
        {
            var points = CalculateROC(scores, classOfInterest);
            return new LCRocResult { Points = points, Auc = CalculateAUC(points) };
        }

        private static List<LCRocDto> CalculateROC(List<LCValidationScore> validationScores, int classOfInterest)
        {
            List<double> thresholds = [];
            if(classOfInterest == 0)
            {
                thresholds = [.. validationScores.Select(p => p.BenignProbability).Distinct().OrderByDescending(x => x)];
            }
            else if(classOfInterest == 1)
            {
                thresholds = [.. validationScores.Select(p => p.MalignantProbability).Distinct().OrderByDescending(x => x)];
            }
            else if(classOfInterest == 2)
            {
                thresholds = [.. validationScores.Select(p => p.NormalProbability).Distinct().OrderByDescending(x => x)];
            }
            else
            {
                throw new ArgumentOutOfRangeException(nameof(classOfInterest));
            }



            var rocDtos = new List<LCRocDto>
            {
                new() { FalsePositiveRate = 0, TruePositiveRate = 0 }
            };
            foreach (var threshold in thresholds)
            {
                int[,] confusionMatrix = CalculateConfusionMatrix(validationScores, threshold, classOfInterest);

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

        private static int[,] CalculateConfusionMatrix(List<LCValidationScore> predictions, double threshold, int classOfInterest)
        {
            int[,] result = new int[2, 2];
            for(int i = 0; i < predictions.Count; i++)
            {
                int trueLabel = predictions[i].TrueLabel;
                int? predictedLabel = null;
                if (classOfInterest == 0)
                {
                    predictedLabel = predictions[i].BenignProbability >= threshold ? 0 : -1;
                }
                else if (classOfInterest == 1)
                {
                    predictedLabel = predictions[i].MalignantProbability >= threshold ? 1 : -1;
                }
                else if (classOfInterest == 2)
                {
                    predictedLabel = predictions[i].NormalProbability >= threshold ? 2 : -1;
                }
                else
                {
                    throw new ArgumentOutOfRangeException(nameof(classOfInterest));
                }

                // true positive
                if (predictedLabel == classOfInterest && trueLabel == classOfInterest)
                {
                    result[0, 0]++;
                }
                // false positive
                else if (predictedLabel == classOfInterest && trueLabel != classOfInterest)
                {
                    result[0, 1]++;
                }
                // false negative
                else if (predictedLabel != classOfInterest && trueLabel == classOfInterest)
                {
                    result[1, 0]++;
                }
                // true negative
                else if (predictedLabel != classOfInterest && trueLabel != classOfInterest)
                {
                    result[1, 1]++;
                }
            }

            return result;
        }
    }
}
