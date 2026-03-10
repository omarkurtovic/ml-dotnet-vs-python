using SharedCL.SentimentAnalysis.Enums;
using SharedCL.Shared.Enums;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL.LungCancerPrediction.Models
{
    public class LungCancerTrainingParams
    {
        public string ModelName { get; set; } = null!;
        public ModelLanguage ModelLanguage { get; set; }
    }
}
