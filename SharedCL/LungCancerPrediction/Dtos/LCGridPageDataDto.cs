using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL
{
    public class LCGridPageDataDto
    {
        public List<LCBasicDto> Models { get; set; } = [];
        public int TotalItems { get; set; }
    }
}
