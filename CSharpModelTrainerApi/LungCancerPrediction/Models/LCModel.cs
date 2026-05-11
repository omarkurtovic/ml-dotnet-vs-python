using CSharpModelTrainerApi.Enums;
using SharedCL;

namespace CSharpModelTrainerApi.LungCancerPrediction.Models
{
    public class LCModel : IEquatable<LCModel>
    {
        public int Id { get; set; }
        public string Name { get; set; } = null!;
        public ModelLanguage Language { get; set; }
        public IList<LCEpochData> EpochData { get; set; } = null!;
        public int TrainingTimeInSeconds { get; set; }
        public string HardwareInfo { get; set; } = null!;

        public bool Equals(LCModel? other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            return Id == other.Id;
        }
        public override bool Equals(object? obj) => obj is LCModel model && Equals(model);

        public override int GetHashCode() => Name.GetHashCode();
        public override string ToString() => Name;
    }
}
