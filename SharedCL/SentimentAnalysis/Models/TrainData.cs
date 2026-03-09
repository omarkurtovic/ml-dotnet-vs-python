using SharedCL.SentimentAnalysis.Enums;
using SharedCL.Shared.Enums;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL.SentimentAnalysis.Models
{
    public class TrainData
    {
        public string ModelName { get; set; } = null!;
        public ModelLanguage ModelLanguage { get; set; }
        public TrainerAlgorithm Algorithm { get; set; } = TrainerAlgorithm.SdcaLogisticRegression;
    }
}

