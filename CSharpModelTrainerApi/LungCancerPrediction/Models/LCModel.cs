using CSharpModelTrainerApi.Enums;
using SharedCL;

namespace CSharpModelTrainerApi.LungCancerPrediction.Models
{
    public class LCModel
    {
        public int Id { get; set; }
        public string Name { get; set; } = null!;
        public ModelLanguage Language { get; set; }
        public int TotalEpochs { get; set; }
        public IList<LCEpochData> EpochData { get; set; } = null!;
        public double TrainingTimeInSeconds { get; set; }
        public double ValidationTimeInSeconds { get; set; }
        public double DataLoadingTimeInSeconds { get; set; }
        public string HardwareInfo { get; set; } = null!;
        public ModelStatus ModelStatus { get; set; } = ModelStatus.Training;
    }
}
