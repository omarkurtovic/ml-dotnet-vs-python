using Azure.Storage.Blobs;
using Microsoft.ML;
using SharedCL.LungCancerPrediction.Models;
using SharedCL.SentimentAnalysis.Models;
using SharedCL.Shared.Enums;
using System.Data;

namespace CSharpModelTrainerApi.Shared
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


        public string GetModelPath(SentimentAnalysisTrainingParams trainParams)
        {
            var repoRoot = GetRepoRoot();
            if (trainParams.ModelLanguage == ModelLanguage.CSharp)
            {
                return Path.Combine(repoRoot, "models", "sentiment-analysis", "csharp", $"{trainParams.ModelName}.zip");
            }
            else if (trainParams.ModelLanguage == ModelLanguage.Python)
            {
                return Path.Combine(repoRoot, "models", "sentiment-analysis", "python", $"{trainParams.ModelName}.onnx");
            }
            else
            {
                return "";
            }
        }

        public string GetModelPath(SentimentAnalysisModel model)
        {
            var repoRoot = GetRepoRoot();
            if (model.Language == ModelLanguage.CSharp)
            {
                return Path.Combine(repoRoot, "models", "sentiment-analysis", "csharp", $"{model.Name}.zip");
            }
            else if (model.Language == ModelLanguage.Python)
            {
                return Path.Combine(repoRoot, "models", "sentiment-analysis", "python", $"{model.Name}.onnx");
            }
            else
            {
                return "";
            }
        }


        public string GetModelPath(LungCancerTrainingParams trainParams)
        {
            var repoRoot = GetRepoRoot();
            if (trainParams.ModelLanguage == ModelLanguage.CSharp)
            {
                return Path.Combine(repoRoot, "models", "lung-cancer-prediction", "csharp", $"{trainParams.ModelName}.weights");
            }
            else if (trainParams.ModelLanguage == ModelLanguage.Python)
            {
                return Path.Combine(repoRoot, "models", "lung-cancer-prediction", "python", $"{trainParams.ModelName}.onnx");
            }
            else
            {
                return "";
            }
        }

        public string GetModelPath(LungCancerModel model)
        {
            var repoRoot = GetRepoRoot();
            if (model.Language == ModelLanguage.CSharp)
            {
                return Path.Combine(repoRoot, "models", "lung-cancer-prediction", "csharp", $"{model.Name}.weights");
            }
            else if (model.Language == ModelLanguage.Python)
            {
                return Path.Combine(repoRoot, "models", "lung-cancer-prediction", "python", $"{model.Name}.onnx");
            }
            else
            {
                return "";
            }
        }

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
    }
}
