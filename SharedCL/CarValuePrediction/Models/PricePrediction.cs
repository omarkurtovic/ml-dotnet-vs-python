using Microsoft.ML.Data;

namespace SharedCL.CarValuePrediction.Models
{
    public class PricePrediction
    {
        [ColumnName("Score")]
        public float Price { get; set; }
    }
}
