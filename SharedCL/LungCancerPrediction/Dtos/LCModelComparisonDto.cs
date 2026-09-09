using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL
{
    public class LCModelComparisonDto
    {   
        public int Id { get; set; }
        public string Name { get; set; } = null!;
        public ModelLanguageDto Language { get; set; }
        public int TotalEpochs { get; set; }
        public IList<LCEpochDataDto> EpochData { get; set; } = null!;
        public TrainingStatusDto ModelStatusDto { get; set; } = TrainingStatusDto.Training;
        public double TrainingTimeInSeconds { get; set; }
        public double ValidationTimeInSeconds { get; set; }
        public double DataLoadingTimeInSeconds { get; set; }
        public string HardwareInfo { get; set; } = null!;

        public bool Equals(LCModelComparisonDto? other)
        {
            if (other is null) return false;
            if (ReferenceEquals(this, other)) return true;
            return Id == other.Id;
        }
        public override bool Equals(object? obj) => obj is LCModelComparisonDto model && Equals(model);

        public override int GetHashCode() => Name.GetHashCode();
        public override string ToString() => Name;
    }
}
