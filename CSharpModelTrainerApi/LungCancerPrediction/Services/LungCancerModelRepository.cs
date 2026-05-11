using CSharpModelTrainerApi.Database;
using Microsoft.EntityFrameworkCore;
using SharedCL.LungCancerPrediction.Dtos;
using SharedCL.LungCancerPrediction.Models;
using SharedCL.Shared.Models;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class LungCancerModelRepository
    {
        private readonly AppDbContext _context;

        public LungCancerModelRepository(AppDbContext context)
        {
            _context = context;
        }
        public async Task<Result> Save(LungCancerModel model)
        {
            _context.LungCancerModels.Add(model);
            await _context.SaveChangesAsync();
            return Result.Success();
        }

        public async Task<Result<LCModelsPageDataDto>> GetModelsPageData(LCModelsGridOptionsDto options)
        {
            var query = _context.LungCancerModels.Include(m => m.EpochData).Where(m => 1 == 1);

            if(!string.IsNullOrWhiteSpace(options.Search))
            {
                query = query.Where(m => m.Name.Contains(options.Search));
            }

            if (!string.IsNullOrWhiteSpace(options.SortBy))
            {
                switch (options.SortBy)
                {
                    case nameof(LCDto.Name):
                        query = options.SortDescending ? query.OrderByDescending(t => t.Name) : query.OrderBy(t => t.Name);
                        break;
                    case nameof(LCDto.Language):
                        query = options.SortDescending ? query.OrderByDescending(t => t.Language) : query.OrderBy(t => t.Language);
                        break;
                    case nameof(LCDto.MacroPrecision):
                        query = options.SortDescending ? query.OrderByDescending(t => t.EpochData.Last().MacroPrecision) : query.OrderBy(t => t.EpochData.Last().MacroPrecision);
                        break;
                    case nameof(LCDto.MacroRecall):
                        query = options.SortDescending ? query.OrderByDescending(t => t.EpochData.Last().MacroRecall) : query.OrderBy(t => t.EpochData.Last().MacroRecall);
                        break;
                    case nameof(LCDto.MacroF1Score):
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

            var models = (await query.ToListAsync()).Select(model => new LCDto()
            {
                Id = model.Id,
                Name = model.Name,
                Language = (ModelLanguageDto)model.Language,
                MacroPrecision = model.EpochData.Last().MacroPrecision,
                MacroRecall = model.EpochData.Last().MacroRecall,
                MacroF1Score = model.EpochData.Last().MacroF1Score
            }).ToList();

            return Result<LCModelsPageDataDto>.Success(new LCModelsPageDataDto()
            {
                Models = models,
                TotalItems = totalItems
            });
        }

        public async Task<Result<LungCancerModel>> GetById(int id)
        {
            var model = await _context.LungCancerModels.Include(m => m.EpochData).FirstAsync();
            if (model == null)
            {
                return Result<LungCancerModel>.Failure("Model not found");
            }
            return Result<LungCancerModel>.Success(model);
        }

        public async Task<Result> Delete(int id)
        {
            var model = await _context.LungCancerModels.FindAsync(id);
            if (model == null)
            {
                return Result.Failure("Model not found");
            }
            _context.LungCancerModels.Remove(model);
            await _context.SaveChangesAsync();
            return Result.Success();
        }
    }
}
