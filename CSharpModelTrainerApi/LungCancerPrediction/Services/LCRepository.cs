using CSharpModelTrainerApi.Database;
using CSharpModelTrainerApi.Enums;
using CSharpModelTrainerApi.LungCancerPrediction.Models;
using Microsoft.EntityFrameworkCore;
using SharedCL;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class LCRepository
    {
        private readonly AppDbContext _context;

        public LCRepository(AppDbContext context)
        {
            _context = context;
        }
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
            var query = _context.LCModels.Include(m => m.EpochData).Where(m => 1 == 1);

            if (!string.IsNullOrWhiteSpace(options.Search))
            {
                query = query.Where(m => m.Name.Contains(options.Search));
            }

            int totalItems = await query.CountAsync();

            if (!string.IsNullOrWhiteSpace(options.SortBy))
            {
                query = options.SortBy switch
                {
                    nameof(LCBasicDto.Name) => options.SortDescending ? query.OrderByDescending(t => t.Name) : query.OrderBy(t => t.Name),
                    nameof(LCBasicDto.Language) => options.SortDescending ? query.OrderByDescending(t => t.Language) : query.OrderBy(t => t.Language),
                    nameof(LCBasicDto.MacroPrecision) => options.SortDescending ? query.OrderByDescending(t => t.EpochData.OrderBy(ed => ed.Epoch).Last().MacroPrecision) : query.OrderBy(t => t.EpochData.OrderBy(ed => ed.Epoch).Last().MacroPrecision),
                    nameof(LCBasicDto.MacroRecall) => options.SortDescending ? query.OrderByDescending(t => t.EpochData.OrderBy(ed => ed.Epoch).Last().MacroRecall) : query.OrderBy(t => t.EpochData.OrderBy(ed => ed.Epoch).Last().MacroRecall),
                    nameof(LCBasicDto.MacroF1Score) => options.SortDescending ? query.OrderByDescending(t => t.EpochData.OrderBy(ed => ed.Epoch).Last().MacroF1Score) : query.OrderBy(t => t.EpochData.OrderBy(ed => ed.Epoch).Last().MacroF1Score),
                    _ => query.OrderByDescending(t => t.Name),
                };
            }
            else
            {
                query = query.OrderByDescending(t => t.Name);
            }


            query = query.Skip(options.CurrentPage * options.PageSize).Take(options.PageSize);
            var models = await query.ToListAsync();
            var modelDtos = models.Select(model => new LCBasicDto()
            {
                Id = model.Id,
                Name = model.Name,
                Language = (ModelLanguageDto)model.Language,
                MacroPrecision = model.EpochData.OrderBy(ed => ed.Epoch).Last().MacroPrecision,
                MacroRecall = model.EpochData.OrderBy(ed => ed.Epoch).Last().MacroRecall,
                MacroF1Score = model.EpochData.OrderBy(ed => ed.Epoch).Last().MacroF1Score,
                Accuracy = model.EpochData.OrderBy(ed => ed.Epoch).Last().ValidationAccuracy,
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
            var model = await _context.LCModels.Where(m => m.Id == id).Include(m => m.EpochData).FirstAsync();
            if (model == null)
            {
                return Result<LCBasicDto>.Failure("Model not found");
            }

            return Result<LCBasicDto>.Success(new LCBasicDto
            {
                Id = model.Id,
                Name = model.Name,
                Language = (ModelLanguageDto)model.Language,
                MacroPrecision = model.EpochData.Last().MacroPrecision,
                MacroRecall = model.EpochData.Last().MacroRecall,
                MacroF1Score = model.EpochData.Last().MacroF1Score,
            });
        }

        public async Task<Result<LCInfoDto>> GetModelInfo(int id)
        {
            var model = await _context.LCModels.Where(m => m.Id == id).Include(m => m.EpochData).FirstAsync();
            if (model == null)
            {
                return Result<LCInfoDto>.Failure("Model not found");
            }
            LCEpochData ed = new LCEpochData();
            int currentEpoch = 0;

            if(model.EpochData != null && model.EpochData.Count != 0)
            {
                ed = model.EpochData.Last();
                currentEpoch = model.EpochData.Count;
            }

            return Result<LCInfoDto>.Success(new LCInfoDto
            {
                TotalEpochs = model.TotalEpochs,
                ModelStatusDto = (ModelStatusDto)model.ModelStatus,
                Name = model.Name,
                TrainingTimeInSeconds = model.TrainingTimeInSeconds,
                HardwareInfo = model.HardwareInfo,
                Language = (ModelLanguageDto)model.Language,
                CurrentEpoch = currentEpoch ,
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
                WeightedF1Score = ed.WeightedF1Score
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

        public async Task<Result> Delete(int id)
        {
            var model = await _context.LCModels.FindAsync(id);
            if (model == null)
            {
                return Result.Failure("Model not found");
            }
            _context.LCModels.Remove(model);
            await _context.SaveChangesAsync();
            return Result.Success();
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
