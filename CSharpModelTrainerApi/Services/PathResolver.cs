using Azure.Storage.Blobs;
using CSharpModelTrainerApi.SentimentAnalysis.Models;
using Microsoft.ML;
using SharedCL;
using SharedCL.LungCancerPrediction.Models;
using System.Data;

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


        public string GetModelPath(SATrainingParamsDto trainParams)
        {
            var repoRoot = GetRepoRoot();
            if (trainParams.Language == ModelLanguage.CSharp)
            {
                return Path.Combine(repoRoot, "models", "sentiment-analysis", "csharp", $"{trainParams.Name}.zip");
            }
            else if (trainParams.Language == ModelLanguage.Python)
            {
                return Path.Combine(repoRoot, "models", "sentiment-analysis", "python", $"{trainParams.Name}.onnx");
            }
            else
            {
                return "";
            }
        }

        public string GetModelPath(SAModel model)
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


        public string GetModelPath(LCTrainingParamsDto trainParams)
        {
            var repoRoot = GetRepoRoot();
            if (trainParams.Language == ModelLanguageDto.CSharp)
            {
                return Path.Combine(repoRoot, "models", "lung-cancer-prediction", "csharp", $"{trainParams.Name}.dat");
            }
            else if (trainParams.Language == ModelLanguageDto.Python)
            {
                return Path.Combine(repoRoot, "models", "lung-cancer-prediction", "python", $"{trainParams.Name}.onnx");
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
                return Path.Combine(repoRoot, "models", "lung-cancer-prediction", "csharp", $"{model.Name}.dat");
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

        public string GetModelPath(LCDto model)
        {
            var repoRoot = GetRepoRoot();
            if (model.Language == ModelLanguageDto.CSharp)
            {
                return Path.Combine(repoRoot, "models", "lung-cancer-prediction", "csharp", $"{model.Name}.dat");
            }
            else if (model.Language == ModelLanguageDto.Python)
            {
                return Path.Combine(repoRoot, "models", "lung-cancer-prediction", "python", $"{model.Name}.onnx");
            }
            else
            {
                return "";
            }
        }

        public string GetModelPath(string modelName, ModelLanguage modelLanguage)
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
