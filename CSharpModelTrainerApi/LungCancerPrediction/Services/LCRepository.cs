using CSharpModelTrainerApi.Database;
using CSharpModelTrainerApi.Enums;
using CSharpModelTrainerApi.LungCancerPrediction.Models;
using Microsoft.EntityFrameworkCore;
using SharedCL;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class LCRepository(AppDbContext context)
    {
        private readonly AppDbContext _context = context;

        public async Task<Result<LCModel>> GetModel(int id, bool withEpoch = false, bool withValidationScores = false)
        {
            if (withValidationScores && !withEpoch)
            {
                throw new Exception("Cannot include validation scores without including epoch data.");
            }

            var query = _context.LCModels.Where(m => m.Id == id);

            if (withValidationScores)
            {
                query = query.Include(m => m.EpochData).ThenInclude(ed => ed.ValidationScores);
            }
            else if (withEpoch)
            {
                query = query.Include(m => m.EpochData);
            }

            var model = await query.FirstOrDefaultAsync();
            if (model == null)
            {
                return Result<LCModel>.Failure("Model not found");
            }
            return Result<LCModel>.Success(model);
        }

        public async Task<Result<List<LCModel>>> GetModels(bool withEpoch, bool withValidationScores)
        {
            var query = _context.LCModels.AsQueryable();

            if (withValidationScores)
            {
                query = query.Include(m => m.EpochData).ThenInclude(ed => ed.ValidationScores);
            }
            else if (withEpoch)
            {
                query = query.Include(m => m.EpochData);
            }

            var models = await query.ToListAsync();
            return Result<List<LCModel>>.Success(models);
        }

        public async Task<Result<List<LCModel>>> GetModelsForOverview(LCGridOptionsDto options)
        {
            var query = _context.LCModels.Include(m => m.EpochData).AsEnumerable();

            if (!string.IsNullOrWhiteSpace(options.Search))
            {
                query = query.Where(m => m.Name.Contains(options.Search));
            }

            int totalItems = query.Count();

            query = options.SortBy switch
            {
                nameof(LCModelOverviewDto.Name) => options.SortDescending ? query.OrderByDescending(t => t.Name) : query.OrderBy(t => t.Name),
                nameof(LCModelOverviewDto.Language) => options.SortDescending ? query.OrderByDescending(t => t.Language) : query.OrderBy(t => t.Language),
                nameof(LCModelOverviewDto.MacroPrecision) => options.SortDescending ? query.OrderByDescending(t => t.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroPrecision ?? 0) : query.OrderBy(t => t.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroPrecision ?? 0),
                nameof(LCModelOverviewDto.MacroRecall) => options.SortDescending ? query.OrderByDescending(t => t.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroRecall ?? 0) : query.OrderBy(t => t.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroRecall ?? 0),
                nameof(LCModelOverviewDto.MacroF1Score) => options.SortDescending ? query.OrderByDescending(t => t.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroF1Score ?? 0) : query.OrderBy(t => t.EpochData.OrderBy(ed => ed.Epoch).LastOrDefault()?.MacroF1Score ?? 0),
                _ => query.OrderByDescending(t => t.Name),
            };


            query = query.Skip(options.CurrentPage * options.PageSize).Take(options.PageSize);
            var models = query.ToList();
            return Result<List<LCModel>>.Success(models);

        }

        public async Task<Result<int>> Save(LCModel model)
        {
            _context.LCModels.Add(model);
            await _context.SaveChangesAsync();
            return Result<int>.Success(model.Id);
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

        public async Task<Result> UpdateStatusAsync(int id, LCTrainingStatus newStatus)
        {
            var model = await _context.LCModels.FindAsync(id);
            if (model == null)
            {
                return Result.Failure("Model not found");
            }
            model.TrainingStatus = newStatus;
            await _context.SaveChangesAsync();
            return Result.Success();
        }

        public async Task<Result> SaveChangesAsync()
        {
            try
            {
                await _context.SaveChangesAsync();
                return Result.Success();
            }
            catch (DbUpdateException ex)
            {
                return Result.Failure(ex.Message);
            }
        }
    }
}
