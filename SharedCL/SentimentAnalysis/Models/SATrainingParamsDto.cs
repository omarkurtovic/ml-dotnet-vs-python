namespace SharedCL
{
    public class SATrainingParamsDto
    {
        public string Name { get; set; } = null!;
        public ModelLanguageDto Language { get; set; }
        public TrainerAlgorithmDto Algorithm { get; set; } = TrainerAlgorithmDto.SdcaLogisticRegression;
    }
}

