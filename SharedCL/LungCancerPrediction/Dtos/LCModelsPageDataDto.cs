using SharedCL.LungCancerPrediction.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL.LungCancerPrediction.Dtos
{
    public class LCModelsPageDataDto
    {
        public List<LCDto> Models { get; set; } = [];
        public int TotalItems { get; set; }
    }
}
