using SharedCL;
using System.Threading.Channels;

namespace CSharpModelTrainerApi.LungCancerPrediction.Workers
{
    public class TrainingQueue
    {
        private readonly Channel<(int, LCTrainingParamsDto)> _channel = Channel.CreateUnbounded<(int, LCTrainingParamsDto)>();
        public ValueTask EnqueueAsync(int modelId, LCTrainingParamsDto trainInfo) => _channel.Writer.WriteAsync((modelId, trainInfo));
        public IAsyncEnumerable<(int, LCTrainingParamsDto)> ReadAllAsync(CancellationToken ct) => _channel.Reader.ReadAllAsync(ct);
    }
}
