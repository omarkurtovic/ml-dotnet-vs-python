using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL
{
    public class LCModelInferenceDto
    {
        public int Id { get; set; }
        public string Name { get; set; } = null!;
        public ModelLanguageDto Language { get; set; }
    }
}
