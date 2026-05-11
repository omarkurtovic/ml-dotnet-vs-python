using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL
{
    public class LCDto
    {
        public int Id { get; set; }
        public string Name { get; set; } = null!;
        public ModelLanguageDto Language { get; set; }
        public double? MacroPrecision { get; set; }
        public double? MacroRecall { get; set; }
        public double? MacroF1Score { get; set; }
    }
}
