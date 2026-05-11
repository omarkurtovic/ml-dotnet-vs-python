using SharedCL.SentimentAnalysis.Enums;
using SharedCL.Shared.Enums;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL.LungCancerPrediction.Dtos
{
    public class LCTrainingParamsDto
    {
        public string Name { get; set; } = null!;
        public ModelLanguageDto Language { get; set; }
        public int Epochs { get; set; }
        public bool WithFlips { get; set; } = true;
    }
}
