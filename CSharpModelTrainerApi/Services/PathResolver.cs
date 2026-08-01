using CSharpModelTrainerApi.Enums;
using SharedCL;
using Microsoft.Extensions.Configuration;

namespace CSharpModelTrainerApi.Services
{
    public class PathResolver
    {
        private readonly string _storageRoot;

        public PathResolver(IConfiguration configuration)
        {
            var configuredRoot = Environment.GetEnvironmentVariable("ML_STORAGE_ROOT")
                ?? configuration["Storage:Root"];

            if (string.IsNullOrWhiteSpace(configuredRoot))
                throw new InvalidOperationException("ML_STORAGE_ROOT is not configured.");

            _storageRoot = Path.GetFullPath(configuredRoot);
        }

        public string GetModelPath(LCDto dto) => GetLCModelPath(dto.Name, (ModelLanguage)dto.Language);

        public string GetModelPath(LCBasicDto dto) => GetLCModelPath(dto.Name, (ModelLanguage)dto.Language);

        public string GetModelPath(LCTrainingParamsDto dto) => GetLCModelPath(dto.Name, (ModelLanguage)dto.Language);

        public string GetLungCancerDataPath()
        {
            return Path.Combine(_storageRoot, "data", "lung-cancer-prediction");
        }

        public string GetLCModelPath(string modelName, ModelLanguage modelLanguage)
        {
            var modelDirectory = modelLanguage switch
            {
                ModelLanguage.CSharp => Path.Combine(_storageRoot, "models", "lung-cancer-prediction", "csharp"),
                ModelLanguage.Python => Path.Combine(_storageRoot, "models", "lung-cancer-prediction", "python"),
                _ => throw new ArgumentOutOfRangeException(nameof(modelLanguage), modelLanguage, "Unsupported model language.")
            };

            Directory.CreateDirectory(modelDirectory);
            return Path.Combine(modelDirectory, $"{modelName}.dat");
        }

        public void InitializeStorage()
        {
            Directory.CreateDirectory(GetLungCancerDataPath());
            Directory.CreateDirectory(Path.Combine(_storageRoot, "models", "lung-cancer-prediction", "csharp"));
            Directory.CreateDirectory(Path.Combine(_storageRoot, "models", "lung-cancer-prediction", "python"));
        }
    }
}
