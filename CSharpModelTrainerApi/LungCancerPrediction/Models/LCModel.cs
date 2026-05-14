using CSharpModelTrainerApi.Enums;
using SharedCL;

namespace CSharpModelTrainerApi.LungCancerPrediction.Models
{
    public class LCModel
    {
        public int Id { get; set; }
        public string Name { get; set; } = null!;
        public ModelLanguage Language { get; set; }
        public IList<LCEpochData> EpochData { get; set; } = null!;
        public int TrainingTimeInSeconds { get; set; }
        public string HardwareInfo { get; set; } = null!;
    }
}
