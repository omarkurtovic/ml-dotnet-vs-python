namespace CSharpModelTrainerApi.LungCancerPrediction.Models
{
    public class LCPredictions
    {
        public int Id { get; set; }
        public int LCEpochDataId { get; set; }
        public LCEpochData LCEpochData { get; set; }
        public double BenignProbability { get; set; }
        public double MalignantProbability { get; set; }
        public double NormalProbability { get; set; }
        public int TrueLabel { get; set; }
    }
}
