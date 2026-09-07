using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL
{
    public class LCValidationScoreDto
    {
        public double BenignProbability { get; set; }
        public double MalignantProbability { get; set; }
        public double NormalProbability { get; set; }
        public int TrueLabel { get; set; }
    }
}
