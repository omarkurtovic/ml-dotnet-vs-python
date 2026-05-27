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

            if (!string.IsNullOrWhiteSpace(options.SortBy))
            {
                switch (options.SortBy)
                {
                    case nameof(LCBasicDto.Name):
                        query = options.SortDescending ? query.OrderByDescending(t => t.Name) : query.OrderBy(t => t.Name);
                        break;
                    case nameof(LCBasicDto.Language):
                        query = options.SortDescending ? query.OrderByDescending(t => t.Language) : query.OrderBy(t => t.Language);
                        break;
                    case nameof(LCBasicDto.MacroPrecision):
                        query = options.SortDescending ? query.OrderByDescending(t => t.EpochData.Last().MacroPrecision) : query.OrderBy(t => t.EpochData.Last().MacroPrecision);
                        break;
                    case nameof(LCBasicDto.MacroRecall):
                        query = options.SortDescending ? query.OrderByDescending(t => t.EpochData.Last().MacroRecall) : query.OrderBy(t => t.EpochData.Last().MacroRecall);
                        break;
                    case nameof(LCBasicDto.MacroF1Score):
                        query = options.SortDescending ? query.OrderByDescending(t => t.EpochData.Last().MacroF1Score) : query.OrderBy(t => t.EpochData.Last().MacroF1Score);
                        break;
                    default:
                        query = query.OrderByDescending(t => t.Name);
                        break;
                }
            }
            else
            {
                query = query.OrderByDescending(t => t.Name);
            }


            int totalItems = await query.CountAsync();
            query = query.Skip(options.CurrentPage * options.PageSize).Take(options.PageSize);

            var models = (await query.ToListAsync()).Select(model => new LCBasicDto()
            {
                Id = model.Id,
                Name = model.Name,
                Language = (ModelLanguageDto)model.Language,
                MacroPrecision = model.EpochData.Last().MacroPrecision,
                MacroRecall = model.EpochData.Last().MacroRecall,
                MacroF1Score = model.EpochData.Last().MacroF1Score
            }).ToList();

            return Result<LCGridPageDataDto>.Success(new LCGridPageDataDto()
            {
                Models = models,
                TotalItems = totalItems
            });
        }

        public async Task<Result<LCDto>> GetModel(int id)
        {
            var model = await _context.LCModels.Where(m => m.Id == id).Include(m => m.EpochData).FirstOrDefaultAsync();
            if(model == null)
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
            if(model == null)
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
            if(model == null)
            {
                return Result<LCInfoDto>.Failure("Model not found");
            }
            return Result<LCInfoDto>.Success(new LCInfoDto
            {
                Name = model.Name,
                NumberOfEpochs = model.EpochData.Count,
                Language = (ModelLanguageDto)model.Language,
                TrainingTimeInSeconds = model.TrainingTimeInSeconds,
                HardwareInfo = model.HardwareInfo,
                TrainingAccuracy = model.EpochData.Last().TrainingAccuracy,
                TrainingLoss = model.EpochData.Last().TrainingLoss,
                ValidationAccuracy = model.EpochData.Last().ValidationAccuracy,
                ValidationLoss = model.EpochData.Last().ValidationLoss,
                BenignPrecision = model.EpochData.Last().BenignPrecision,
                BenignRecall = model.EpochData.Last().BenignRecall,
                BenignF1Score = model.EpochData.Last().BenignF1Score,
                MalignantPrecision = model.EpochData.Last().MalignantPrecision,
                MalignantRecall = model.EpochData.Last().MalignantRecall,
                MalignantF1Score = model.EpochData.Last().MalignantF1Score,
                NormalPrecision = model.EpochData.Last().NormalPrecision,
                NormalRecall = model.EpochData.Last().NormalRecall,
                NormalF1Score = model.EpochData.Last().NormalF1Score,
                MacroPrecision = model.EpochData.Last().MacroPrecision,
                MacroRecall = model.EpochData.Last().MacroRecall,
                MacroF1Score = model.EpochData.Last().MacroF1Score,
                WeightedPrecision = model.EpochData.Last().WeightedPrecision,
                WeightedRecall = model.EpochData.Last().WeightedRecall,
                WeightedF1Score = model.EpochData.Last().WeightedF1Score
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
    }
}
