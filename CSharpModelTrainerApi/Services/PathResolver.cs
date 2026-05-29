using CSharpModelTrainerApi.Enums;
using SharedCL;

namespace CSharpModelTrainerApi.Services
{
    public class PathResolver
    {
        private static string GetRepoRoot()
        {
            var envRoot = Environment.GetEnvironmentVariable("REPO_ROOT");
            if (!string.IsNullOrEmpty(envRoot))
                return envRoot;
            return Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));
        }

        public string GetModelPath(LCDto dto) => GetLCModelPath(dto.Name, (ModelLanguage)dto.Language);
        
        public string GetModelPath(LCBasicDto dto) => GetLCModelPath(dto.Name, (ModelLanguage)dto.Language);

        public string GetModelPath(LCTrainingParamsDto dto) => GetLCModelPath(dto.Name, (ModelLanguage)dto.Language);

        public string GetLungCancerDataPath()
        {
            var repoRoot = GetRepoRoot();
            return Path.Join(repoRoot, "data", "lung-cancer-prediction");
        }

        public string GetLCModelPath(string modelName, ModelLanguage modelLanguage)
        {
            var repoRoot = GetRepoRoot();
            if (modelLanguage == ModelLanguage.CSharp)
            {
                return Path.Combine(repoRoot, "models", "lung-cancer-prediction", "csharp", $"{modelName}.dat");
            }
            else if (modelLanguage == ModelLanguage.Python)
            {
                return Path.Combine(repoRoot, "models", "lung-cancer-prediction", "python", $"{modelName}.onnx");
            }
            else
            {
                return "";
            }
        }
    }
}
