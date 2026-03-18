using Microsoft.AspNetCore.Mvc.ViewEngines;
using CSharpModelTrainerApi.Shared;
using Microsoft.ML;
using SharedCL.SentimentAnalysis.Mappings;
using SharedCL.SentimentAnalysis.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SharedCL.Shared.Models;

namespace CSharpModelTrainerApi.SentimentAnalysis.Services
{
    public class SentimentAnalysisPredictionServices(BlobService blobService)
    {
        public async Task<SentimentPrediction> Predict(SentimentAnalysisModel model, string review)
        {
            if (model.Language == SharedCL.Shared.Enums.ModelLanguage.CSharp)
                return await PredictWithMlNet(model, review);
            else if (model.Language == SharedCL.Shared.Enums.ModelLanguage.Python)
                return await PredictWithOnnx(model, review);

            return new SentimentPrediction();
        }

        private async Task<SentimentPrediction> PredictWithMlNet(SentimentAnalysisModel model, string review)
        {
            MLContext mlContext = new MLContext();
            var modelPath = await blobService.EnsureModelDownloadedAsync($"sentiment-analysis/csharp/{model.Name}.zip");

            mlContext.ComponentCatalog.RegisterAssembly(typeof(SentimentCleanerMapping).Assembly);
            ITransformer trainedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

            var sentimentData = new SentimentData() { Review = review };
            var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(trainedModel);
            return predictionEngine.Predict(sentimentData);
        }

        private async Task<SentimentPrediction> PredictWithOnnx(SentimentAnalysisModel model, string review)
        {
            var modelPath = await blobService.EnsureModelDownloadedAsync($"sentiment-analysis/python/{model.Name}.onnx");

            var features = new string[] { review };

            using var session = new InferenceSession(modelPath);
            var inputName = session.InputMetadata.Keys.First();
            var tensor = new DenseTensor<string>(features, [1, features.Length]);
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensor) };

            using var results = session.Run(inputs);
            NamedOnnxValue label = results[0];
            NamedOnnxValue prediction = results[1];

            var result = new SentimentPrediction();
            result.IsPositive = label.Value is IEnumerable<long> labelValues && labelValues.FirstOrDefault() == 1;
            result.Probability = ((prediction.Value as IEnumerable<DisposableNamedOnnxValue>)!.First().Value as Dictionary<long, float>)![1];

            return result;
        }
    }
}
