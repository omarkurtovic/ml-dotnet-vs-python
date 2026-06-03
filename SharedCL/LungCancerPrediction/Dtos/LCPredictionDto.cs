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

        public double PredictionScore
        {
            get
            {
                if (BenignScore >= MalignantScore && BenignScore >= NormalScore)
                    return Math.Round(BenignScore * 100, 2);
                else if (MalignantScore >= BenignScore && MalignantScore >= NormalScore)
                    return Math.Round(MalignantScore * 100, 2);
                else
                    return Math.Round(NormalScore * 100, 2);
            }
        }
    }
}
