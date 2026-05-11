using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL
{
    public class SAModelDto
    {
        public int Id { get; set; }
        public string Name { get; set; } = null!;
        public ModelLanguageDto Language { get; set; }
        public TrainerAlgorithmDto TrainerAlgorithm { get; set; }
        public double? TrainingAccuracy { get; set; }
        public double? TrainingF1Score { get; set; }
        public double? TrainingAreaUnderRocCurve { get; set; }
        public double? TrainingPositivePrecision { get; set; }
        public double? TrainingPositiveRecall { get; set; }
        public double? TestingAccuracy { get; set; }
        public double? TestingF1Score { get; set; }
        public double? TestingAreaUnderRocCurve { get; set; }
        public double? TestingPositivePrecision { get; set; }
        public double? TestingPositiveRecall { get; set; }
    }
}
