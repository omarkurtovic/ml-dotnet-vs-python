using SharedCL.Shared.Enums;

namespace SharedCL.LungCancerPrediction.Models
{
    public class LungCancerModel
    {
        public int Id { get; set; }
        public string Name { get; set; } = null!;
        public ModelLanguage Language { get; set; }

        public double? TrainingAccuracy { get; set; }
        public double? ValidationAccuracy { get; set; }
        public double? ValidationLoss { get; set; }

        public double? BenignPrecision { get; set; }
        public double? BenignRecall { get; set; }
        public double? BenignF1Score { get; set; }

        public double? MalignantPrecision { get; set; }
        public double? MalignantRecall { get; set; }
        public double? MalignantF1Score { get; set; }

        public double? NormalPrecision { get; set; }
        public double? NormalRecall { get; set; }
        public double? NormalF1Score { get; set; }

        public double? MacroPrecision { get; set; }
        public double? MacroRecall { get; set; }
        public double? MacroF1Score { get; set; }

        public double? WeightedPrecision { get; set; }
        public double? WeightedRecall { get; set; }
        public double? WeightedF1Score { get; set; }
    }
}
