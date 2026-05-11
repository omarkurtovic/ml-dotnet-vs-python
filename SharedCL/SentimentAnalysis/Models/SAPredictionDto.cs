using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL
{
    public class SAPredictionDto
    {
        [ColumnName("PredictedLabel")]
        public bool IsPositive { get; set; }

        [ColumnName("Probability")]
        public float Probability { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }
    }
}
