using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL.LungCancerPrediction.Dtos
{
    public class LCWithEpochs : LCDto
    {
        public int TrainingTimeInSeconds { get; set; }
        public string HardwareInfo { get; set; } = null!;
        public int NumberOfEpochs { get; set; }
        public List<LCEpochDataDto> Epochs { get; set; } = [];
    }
}
