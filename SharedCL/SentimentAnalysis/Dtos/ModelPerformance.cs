using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL.SentimentAnalysis.Dtos
{
    public class ModelPerformance
    {
        public double TrainingAccuracy { get; set; }
        public double TrainingF1Score { get; set; }
        public double TrainingAreaUnderRocCurve { get; set; }
        public double TrainingPositivePrecision { get; set; }
        public double TrainingPositiveRecall { get; set; }

        public double TestingAccuracy { get; set; }
        public double TestingF1Score { get; set; }
        public double TestingAreaUnderRocCurve { get; set; }
        public double TestingPositivePrecision { get; set; }
        public double TestingPositiveRecall { get; set; }

    }
}
