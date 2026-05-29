namespace SharedCL
{
    public class LCPredictionDto
    {
        public float BenignScore { get; set; }
        public float MalignantScore { get; set; }
        public float NormalScore { get; set; }
        public double PredictionTimeInSeconds { get; set; }
        public PredictionTypeDto PredictionType
        {
            get
            {
                if (BenignScore >= MalignantScore && BenignScore >= NormalScore)
                    return PredictionTypeDto.Benign;
                else if (MalignantScore >= BenignScore && MalignantScore >= NormalScore)
                    return PredictionTypeDto.Malignant;
                else
                    return PredictionTypeDto.Normal;
            }
        }
    }
}
