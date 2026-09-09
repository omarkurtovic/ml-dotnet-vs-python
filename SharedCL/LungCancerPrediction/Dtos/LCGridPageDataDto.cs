using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL
{
    public class LCGridPageDataDto
    {
        public List<LCModelOverviewDto> Models { get; set; } = [];
        public int TotalItems { get; set; }
    }
}
