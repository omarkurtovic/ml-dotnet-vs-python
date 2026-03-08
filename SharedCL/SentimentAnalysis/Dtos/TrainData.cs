using SharedCL.Shared.Enums;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL.SentimentAnalysis.Dtos
{
    public class TrainData
    {
        public string ModelName { get; set; }
        public ModelLanguage ModelLanguage { get; set; }
    }
}
