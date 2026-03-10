using CSharpModelTrainerApi.Database;
using Microsoft.EntityFrameworkCore;
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

        public async Task<Result<List<LungCancerModel>>> GetAll()
        {
            var models = await _context.LungCancerModels.ToListAsync();
            return Result<List<LungCancerModel>>.Success(models);
        }

        public async Task<Result<LungCancerModel>> GetById(int id)
        {
            var model = await _context.LungCancerModels.FindAsync(id);
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
