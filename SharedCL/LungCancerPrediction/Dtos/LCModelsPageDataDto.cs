using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL
{
    public class LCModelsPageDataDto
    {
        public List<LCDto> Models { get; set; } = [];
        public int TotalItems { get; set; }
    }
}
