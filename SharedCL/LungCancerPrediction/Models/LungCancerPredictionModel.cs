namespace SharedCL.LungCancerPrediction.Models
{
    public class LungCancerPredictionModel
    {
        public float BenignScore { get; set; }
        public float MalignantScore { get; set; }
        public float NormalScore { get; set; }

        public string PredictedLabel => new[] {
            ("Benign", BenignScore),
            ("Malignant", MalignantScore),
            ("Normal", NormalScore)
        }.MaxBy(x => x.Item2).Item1;
    }
}
