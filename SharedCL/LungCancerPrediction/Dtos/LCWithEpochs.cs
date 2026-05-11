using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL.LungCancerPrediction.Dtos
{
    public class LCWithEpochs : LCDto
    {
        public List<LCEpochDataDto> Epochs { get; set; } = [];
    }
}
