using SharedCL.Shared.Enums;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL.Shared.Models
{
    public class MLModel
    {
        public string Name { get; set; } = null!;
        public string Description { get; set; } = null!;
        public ModelLanguage Language { get; set; }
    }
}
