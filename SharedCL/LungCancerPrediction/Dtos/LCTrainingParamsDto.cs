using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL
{
    public class LCTrainingParamsDto
    {
        public string Name { get; set; } = null!;
        public ModelLanguageDto Language { get; set; }
        public int Epochs { get; set; }
        public bool WithTrasforms { get; set; } = true;
    }
}
