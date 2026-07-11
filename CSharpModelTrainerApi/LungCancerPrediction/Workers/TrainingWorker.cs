using CSharpModelTrainerApi.Database;
using CSharpModelTrainerApi.LungCancerPrediction.Services;
using CSharpModelTrainerApi.Services;

namespace CSharpModelTrainerApi.LungCancerPrediction.Workers
{
    public class TrainingWorker(TrainingQueue queue, IServiceScopeFactory scopeFactory) : BackgroundService
    {
        private readonly TrainingQueue _queue = queue;
        private readonly IServiceScopeFactory _scopeFactory = scopeFactory;

        protected override async Task ExecuteAsync(CancellationToken ct)
        {
            await foreach (var (modelId, trainInfo) in _queue.ReadAllAsync(ct))
            {
                using var scope = _scopeFactory.CreateScope();
                var lcRepository = scope.ServiceProvider.GetRequiredService<LCRepository>();
                var pathResolver = scope.ServiceProvider.GetRequiredService<PathResolver>();
                
                await new LCTrainer(pathResolver, lcRepository).TrainModelAsync(modelId, trainInfo);
            }
        }
    }
}
