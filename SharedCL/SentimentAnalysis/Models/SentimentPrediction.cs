using Microsoft.ML.Data;

namespace SharedCL.SentimentAnalysis.Models
{
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsPositive { get; set; }

        [ColumnName("Probability")]
        public float Probability { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }
    }
}

