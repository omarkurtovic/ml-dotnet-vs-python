using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL.SentimentAnalysis.Models
{
    public class SentimentData
    {
        [LoadColumn(0)]
        public string Review { get; set; } = null!;

        [LoadColumn(1)]
        public string Sentiment { get; set; } = null!;
    }
}
