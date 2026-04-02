using SharedCL.Shared.Enums;

namespace SharedCL.LungCancerPrediction.Models
{
    public class LungCancerModel
    {
        public int Id { get; set; }
        public string Name { get; set; } = null!;
        public ModelLanguage Language { get; set; }
        public IList<LungCancerModelEpochData> EpochData { get; set; } = null!;

    }
}
