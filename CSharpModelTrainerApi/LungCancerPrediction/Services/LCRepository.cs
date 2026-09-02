using CSharpModelTrainerApi.Database;
using CSharpModelTrainerApi.Enums;
using CSharpModelTrainerApi.LungCancerPrediction.Models;
using Microsoft.EntityFrameworkCore;
using SharedCL;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class LCRepository(AppDbContext context, ROCService rocService)
    {
        private readonly AppDbContext _context = context;
        private readonly ROCService _rocService = rocService;

        public async Task<Result<List<LCDto>>> GetModels()
        {
            var models = await _context.LCModels.Include(m => m.EpochData).ToListAsync();
            return Result<List<LCDto>>.Success([.. models.Select(model => new LCDto()
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
            })]);
        }
        public async Task<Result<List<LCBasicDto>>> GetModelsBasic()
        {
            return Result<List<LCBasicDto>>.Success([.. _context.LCModels.Select(model => new LCBasicDto()
            {
                Id = model.Id,
                Name = model.Name,
                Language = (ModelLanguageDto)model.Language,
                TrainingTimeInSeconds = model.TrainingTimeInSeconds,
                HardwareInfo = model.HardwareInfo
            })]);
        }

        public async Task<Result<LCGridPageDataDto>> GetModelsSearch(LCGridOptionsDto options)
        {
            var query = _context.LCModels.Include(m => m.EpochData).AsEnumerable();

            if (!string.IsNullOrWhiteSpace(options.Search))
            {
                query = query.Where(m => m.Name.Contains(options.Search));
            }

            int totalItems = query.Count();

            query = options.SortBy switch
            {
                nameof(LCBasicDto.Name) => options.SortDescending ? query.OrderByDescending(t => t.Name) : query.OrderBy(t => t.Name),
                nameof(LCBasicDto.Language) => options.SortDescending ? query.OrderByDescending(t => t.Language) : query.OrderBy(t => t.Language),
                nameof(LCBasicDto.MacroPrecision) => options.SortDescending ? query.OrderByDescending(t => t.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroPrecision ?? 0) : query.OrderBy(t => t.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroPrecision ?? 0),
                nameof(LCBasicDto.MacroRecall) => options.SortDescending ? query.OrderByDescending(t => t.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroRecall ?? 0) : query.OrderBy(t => t.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroRecall ?? 0),
                nameof(LCBasicDto.MacroF1Score) => options.SortDescending ? query.OrderByDescending(t => t.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroF1Score ?? 0) : query.OrderBy(t => t.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroF1Score ?? 0),
                _ => query.OrderByDescending(t => t.Name),
            };


            query = query.Skip(options.CurrentPage * options.PageSize).Take(options.PageSize);
            var models = query.ToList();
            var modelDtos = models.Select(model => new LCBasicDto()
            {
                Id = model.Id,
                Name = model.Name,
                Language = (ModelLanguageDto)model.Language,
                MacroPrecision = model.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroPrecision ?? 0,
                MacroRecall = model.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroRecall ?? 0,
                MacroF1Score = model.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroF1Score ?? 0,
                Accuracy = model.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.ValidationAccuracy ?? 0,
            }).ToList();

            return Result<LCGridPageDataDto>.Success(new LCGridPageDataDto()
            {
                Models = modelDtos,
                TotalItems = totalItems
            });
        }

        public async Task<Result<LCDto>> GetModel(int id)
        {
            var model = await _context.LCModels.Where(m => m.Id == id).Include(m => m.EpochData).FirstOrDefaultAsync();
            if (model == null)
            {
                return Result<LCDto>.Failure("Model not found");
            }

            return Result<LCDto>.Success(new LCDto()
            {
                Id = model.Id,
                Name = model.Name,
                Language = (ModelLanguageDto)model.Language,
                TrainingTimeInSeconds = model.TrainingTimeInSeconds,
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
            });
        }

        public async Task<Result<LCBasicDto>> GetModelBasic(int id)
        {
            var model = await _context.LCModels.Where(m => m.Id == id).Include(m => m.EpochData).FirstOrDefaultAsync();
            if (model == null)
            {
                return Result<LCBasicDto>.Failure("Model not found");
            }

            if (model.EpochData == null || model.EpochData.Count == 0)
            {
                return Result<LCBasicDto>.Success(new LCBasicDto
                {
                    Id = model.Id,
                    Name = model.Name,
                    Language = (ModelLanguageDto)model.Language,
                });
            }

            var lastEpoch = model.EpochData.OrderBy(ed => ed.Epoch).Last();
            return Result<LCBasicDto>.Success(new LCBasicDto
            {
                Id = model.Id,
                Name = model.Name,
                Language = (ModelLanguageDto)model.Language,
                MacroPrecision = lastEpoch.MacroPrecision,
                MacroRecall = lastEpoch.MacroRecall,
                MacroF1Score = lastEpoch.MacroF1Score,
            });
        }

        public async Task<Result<LCInfoDto>> GetModelInfo(int id)
        {
            var model = await _context.LCModels.Where(m => m.Id == id).Include(m => m.EpochData)
                .ThenInclude(m => m.LCPredictions).FirstOrDefaultAsync();
            if (model == null)
            {
                return Result<LCInfoDto>.Failure("Model not found");
            }
            LCEpochData ed = new();
            int currentEpoch = 0;

            if (model.EpochData != null && model.EpochData.Count != 0)
            {
                ed = model.EpochData.Last();
                currentEpoch = model.EpochData.Count;
            }

            var rocData = _rocService.CalculateROC(ed.LCPredictions);
            return Result<LCInfoDto>.Success(new LCInfoDto
            {
                TotalEpochs = model.TotalEpochs,
                ModelStatusDto = (ModelStatusDto)model.ModelStatus,
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
                RocData = rocData,
                AUC = _rocService.CalculateAUC(rocData)
            });
        }



        public async Task<Result<int>> Save(LCDto model)
        {
            var dbModel = new LCModel
            {
                Name = model.Name,
                Language = (ModelLanguage)model.Language,
                TrainingTimeInSeconds = model.TrainingTimeInSeconds,
                HardwareInfo = model.HardwareInfo,
                TotalEpochs = model.TotalEpochs,
                ModelStatus = (ModelStatus)model.ModelStatusDto,
                EpochData = [.. model.EpochData.Select(epoch => new LCEpochData
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
            };

            _context.LCModels.Add(dbModel);
            await _context.SaveChangesAsync();
            return Result<int>.Success(dbModel.Id);
        }

        public async Task<Result> Delete(int id, Action deleteModelFile)
        {
            var model = await _context.LCModels.FindAsync(id);
            if (model == null)
            {
                return Result.Failure("Model not found");
            }

            await using var transaction = await _context.Database.BeginTransactionAsync();
            try
            {
                _context.LCModels.Remove(model);
                await _context.SaveChangesAsync();

                deleteModelFile();

                await transaction.CommitAsync();
                return Result.Success();
            }
            catch (Exception ex)
            {
                await transaction.RollbackAsync();
                return Result.Failure($"Greška prilikom brisanja modela: {ex.Message}");
            }
        }

        public async Task<Result> UpdateNameAsync(int id, string newName)
        {
            var model = await _context.LCModels.FindAsync(id);
            if (model == null)
            {
                return Result.Failure("Model not found");
            }
            model.Name = newName;
            await _context.SaveChangesAsync();
            return Result.Success();
        }

        public async Task<Result> UpdateStatusAsync(int id, ModelStatus newStatus)
        {
            var model = await _context.LCModels.FindAsync(id);
            if (model == null)
            {
                return Result.Failure("Model not found");
            }
            model.ModelStatus = newStatus;
            await _context.SaveChangesAsync();
            return Result.Success();
        }

        public async Task<Result> UpdateTrainingTimeAsync(int id, double trainingTimeInSeconds)
        {
            var model = await _context.LCModels.FindAsync(id);
            if (model == null)
            {
                return Result.Failure("Model not found");
            }
            model.TrainingTimeInSeconds = trainingTimeInSeconds;
            await _context.SaveChangesAsync();
            return Result.Success();
        }

        public async Task<Result> AddEpochData(int id, LCEpochData epoch)
        {
            var model = await _context.LCModels.FindAsync(id);
            if (model == null)
            {
                return Result.Failure("Model not found");
            }
            model.EpochData.Add(epoch);
            await _context.SaveChangesAsync();
            return Result.Success();
        }

        public async Task<Result> AddEpochData(int id, LCEpochDataDto epoch)
        {
            var model = await _context.LCModels.FindAsync(id);
            if (model == null)
            {
                return Result.Failure("Model not found");
            }
            var newEpochData = new LCEpochData
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
            };
            model.EpochData.Add(newEpochData);
            await _context.SaveChangesAsync();
            return Result.Success();
        }
    }
}
