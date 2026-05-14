using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL
{
    public class LCDto : IEquatable<LCDto>
    {
        public int Id { get; set; }
        public string Name { get; set; } = null!;
        public ModelLanguageDto Language { get; set; }
        public IList<LCEpochDataDto> EpochData { get; set; } = null!;
        public int TrainingTimeInSeconds { get; set; }
        public string HardwareInfo { get; set; } = null!;

        public bool Equals(LCDto? other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            return Id == other.Id;
        }
        public override bool Equals(object? obj) => obj is LCDto model && Equals(model);

        public override int GetHashCode() => Name.GetHashCode();
        public override string ToString() => Name;

    }
}
