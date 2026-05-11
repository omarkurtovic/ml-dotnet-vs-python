using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL.LungCancerPrediction.Dtos
{
    public class LCModelsGridOptionsDto
    {
        public bool IsReoccuring { get; set; } = false;
        public int PageSize { get; set; } = 10;
        public int CurrentPage { get; set; } = 0;
        public string? SortBy { get; set; }
        public bool SortDescending { get; set; } = false;
        public string? Search { get; set; } = null!;
    }
}
