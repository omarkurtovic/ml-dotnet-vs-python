using CSharpModelTrainerApi.LungCancerPrediction.Models;
using CSharpModelTrainerApi.LungCancerPrediction.Services;
using SharedCL;

namespace CSharpModelTrainerApi.Mappers
{
    public static class EntityToDtoMapper
    {
        public static LCInfoDto ToInfoDto(this LCModel model, LCRocResult rocResult)
        {
            LCEpochData ed = new();
            int currentEpoch = 0;

            if (model.EpochData != null && model.EpochData.Count != 0)
            {
                ed = model.EpochData.Last();
                currentEpoch = model.EpochData.Count;
            }

            return new LCInfoDto
            {
                TotalEpochs = model.TotalEpochs,
                ModelStatusDto = (TrainingStatusDto)model.TrainingStatus,
                Name = model.Name,
                TrainingTimeInSeconds = model.TrainingTimeInSeconds,
                HardwareInfo = model.HardwareInfo,
                Language = (ModelLanguageDto)model.Language,
                CurrentEpoch = currentEpoch,
                TrainingAccuracy = ed.TrainingAccuracy,
                TrainingLoss = ed.TrainingLoss,
                ValidationAccuracy = ed.ValidationAccuracy,
                ValidationLoss = ed.ValidationLoss,
                BenignPrecision = ed.BenignPrecision,
                BenignRecall = ed.BenignRecall,
                BenignF1Score = ed.BenignF1Score,
                MalignantPrecision = ed.MalignantPrecision,
                MalignantRecall = ed.MalignantRecall,
                MalignantF1Score = ed.MalignantF1Score,
                NormalPrecision = ed.NormalPrecision,
                NormalRecall = ed.NormalRecall,
                NormalF1Score = ed.NormalF1Score,
                MacroPrecision = ed.MacroPrecision,
                MacroRecall = ed.MacroRecall,
                MacroF1Score = ed.MacroF1Score,
                WeightedPrecision = ed.WeightedPrecision,
                WeightedRecall = ed.WeightedRecall,
                WeightedF1Score = ed.WeightedF1Score,
                RocData = rocResult.Points,
                AUC = rocResult.Auc
            };
        }

        public static List<LCModelComparisonDto> ToComparisonDto(this IEnumerable<LCModel> models)
        {
            return [.. models.Select(model => new LCModelComparisonDto()
            {
                Id = model.Id,
                Name = model.Name,
                Language = (ModelLanguageDto)model.Language,
                TrainingTimeInSeconds = model.TrainingTimeInSeconds,
                ValidationTimeInSeconds = model.ValidationTimeInSeconds,
                DataLoadingTimeInSeconds = model.DataLoadingTimeInSeconds,
                HardwareInfo = model.HardwareInfo,
                EpochData = [.. model.EpochData.Select(epoch => new LCEpochDataDto
                {
                    Epoch = epoch.Epoch,
                    TrainingLoss = epoch.TrainingLoss,
                    TrainingAccuracy = epoch.TrainingAccuracy,
                    ValidationLoss = epoch.ValidationLoss,
                    ValidationAccuracy = epoch.ValidationAccuracy,
                    BenignPrecision = epoch.BenignPrecision,
                    BenignRecall = epoch.BenignRecall,
                    BenignF1Score = epoch.BenignF1Score,
                    MalignantPrecision = epoch.MalignantPrecision,
                    MalignantRecall = epoch.MalignantRecall,
                    MalignantF1Score = epoch.MalignantF1Score,
                    NormalPrecision = epoch.NormalPrecision,
                    NormalRecall = epoch.NormalRecall,
                    NormalF1Score = epoch.NormalF1Score,
                    MacroPrecision = epoch.MacroPrecision,
                    MacroRecall = epoch.MacroRecall,
                    MacroF1Score = epoch.MacroF1Score,
                    WeightedPrecision = epoch.WeightedPrecision,
                    WeightedRecall = epoch.WeightedRecall,
                    WeightedF1Score = epoch.WeightedF1Score
                })]
            })];
        }

        public static LCGridPageDataDto ToOverviewDto(this IEnumerable<LCModel> models)
        {
            var modelDtos = models.Select(model => new LCModelOverviewDto()
            {
                Id = model.Id,
                Name = model.Name,
                Language = (ModelLanguageDto)model.Language,
                MacroPrecision = model.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroPrecision ?? 0,
                MacroRecall = model.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroRecall ?? 0,
                MacroF1Score = model.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroF1Score ?? 0,
                Accuracy = model.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.ValidationAccuracy ?? 0,
            }).ToList();

            return new LCGridPageDataDto()
            {
                Models = modelDtos,
                TotalItems = models.Count()
            };
        }

        public static List<LCModelInferenceDto> ToInferenceDto(this IEnumerable<LCModel> models)
        {
            return [..models.Select(model => new LCModelInferenceDto()
            {
                Id = model.Id,
                Name = model.Name,
                Language = (ModelLanguageDto)model.Language
            })];
        }

        
    }
}
