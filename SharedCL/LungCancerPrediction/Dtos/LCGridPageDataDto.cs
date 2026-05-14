using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL
{
    public class LCGridPageDataDto
    {
        public List<LCBasic> Models { get; set; } = new List<LCBasic>();
        public int TotalItems { get; set; }
    }
}
