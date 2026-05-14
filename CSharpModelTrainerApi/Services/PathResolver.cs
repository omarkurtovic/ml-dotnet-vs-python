using CSharpModelTrainerApi.Enums;
using SharedCL;

namespace CSharpModelTrainerApi.Services
{
    public class PathResolver
    {
        private string GetRepoRoot()
        {
            var envRoot = Environment.GetEnvironmentVariable("REPO_ROOT");
            if (!string.IsNullOrEmpty(envRoot))
                return envRoot;
            return Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));
        }

        public string GetModelPath(LCDto dto) => GetLCModelPath(dto.Name, (ModelLanguage)dto.Language, ModelType.LungCancer);
        
        public string GetModelPath(LCBasic dto) => GetLCModelPath(dto.Name, (ModelLanguage)dto.Language, ModelType.LungCancer);

        public string GetModelPath(LCInfo dto) => GetLCModelPath(dto.Name, (ModelLanguage)dto.Language, ModelType.LungCancer);

        public string GetModelPath(SADto dto) => GetLCModelPath(dto.Name, (ModelLanguage)dto.Language, ModelType.SentimentAnalysis);

        public string GetModelPath(LCTrainingParamsDto dto) => GetLCModelPath(dto.Name, (ModelLanguage)dto.Language, ModelType.SentimentAnalysis);

        public string GetSentimentDataPath()
        {
            var repoRoot = GetRepoRoot();
            return Path.Combine(repoRoot, "data", "sentiment-analysis", "IMDB Dataset.csv");
        }

        public string GetLungCancerDataPath()
        {
            var repoRoot = GetRepoRoot();
            return Path.Join(repoRoot, "data", "lung-cancer-prediction");
        }

        public string GetLCModelPath(string modelName, ModelLanguage modelLanguage, ModelType modelType)
        {
            var repoRoot = GetRepoRoot();
            var modelFolder = modelType == ModelType.SentimentAnalysis ? "sentiment-analysis" : "lung-cancer-prediction";
            if (modelLanguage == ModelLanguage.CSharp)
            {
                return Path.Combine(repoRoot, "models", modelFolder, "csharp", $"{modelName}.dat");
            }
            else if (modelLanguage == ModelLanguage.Python)
            {
                return Path.Combine(repoRoot, "models", modelFolder, "python", $"{modelName}.onnx");
            }
            else
            {
                return "";
            }
        }
    }
}
