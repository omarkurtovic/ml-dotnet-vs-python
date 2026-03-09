using CSharpModelTrainerApi.Database;
using Microsoft.EntityFrameworkCore;
using SharedCL.SentimentAnalysis.Models;
using SharedCL.Shared.Models;

namespace CSharpModelTrainerApi.SentimentAnalysis.Services
{
    public class SentimentAnalysisRepository
    {
        private readonly AppDbContext _context;

        public SentimentAnalysisRepository(AppDbContext context)
        {
            _context = context;
        }
        public async Task<Result> Save(SentimentAnalysisModel model)
        {
            _context.SentimentAnalysisModels.Add(model);
            await _context.SaveChangesAsync();
            return Result.Success();
        }

        public async Task<Result<List<SentimentAnalysisModel>>> GetAll()
        {
            var models = await _context.SentimentAnalysisModels.ToListAsync();
            return Result<List<SentimentAnalysisModel>>.Success(models);
        }

        public async Task<Result<SentimentAnalysisModel>> GetById(int id)
        {
            var model = await _context.SentimentAnalysisModels.FindAsync(id);
            if (model == null)
            {
                return Result<SentimentAnalysisModel>.Failure("Model not found");
            }
            return Result<SentimentAnalysisModel>.Success(model);
        }

        public async Task<Result> Delete(int id)
        {
            var model = await _context.SentimentAnalysisModels.FindAsync(id);
            if (model == null)
            {
                return Result.Failure("Model not found");
            }
            _context.SentimentAnalysisModels.Remove(model);
            await _context.SaveChangesAsync();
            return Result.Success();
        }
    }
}
