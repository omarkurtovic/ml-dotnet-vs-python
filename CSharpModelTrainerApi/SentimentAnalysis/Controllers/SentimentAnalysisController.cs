using CSharpModelTrainerApi.SentimentAnalysis.Services;
using Microsoft.AspNetCore.Mvc;
using SharedCL.SentimentAnalysis.Dtos;
using SharedCL.SentimentAnalysis.Models;
using SharedCL.Shared.Enums;
using SharedCL.Shared.Models;

namespace CSharpModelTrainerApi.SentimentAnalysis.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class SentimentAnalysisController : ControllerBase
    {

        [HttpGet]
        [Route("GetModels")]
        public IActionResult GetModels()
        {
            var repoRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));
            var csharpModelsPath = Path.Combine(repoRoot, "models", "sentiment-analysis", "csharp");
            var pythonModelsPath = Path.Combine(repoRoot, "models", "sentiment-analysis", "python");

            var result = new List<MLModel>();
            foreach(var file in Directory.GetFiles(csharpModelsPath, "*.zip"))
            {
                var modelName = Path.GetFileNameWithoutExtension(file);
                result.Add(new MLModel
                {
                    Name = modelName,
                    Description = $"A sentiment analysis model trained using C# and ML.NET. File: {file}",
                    Language = ModelLanguage.CSharp
                });
            }
            foreach(var file in Directory.GetFiles(pythonModelsPath, "*.onnx"))
            {
                var modelName = Path.GetFileNameWithoutExtension(file);
                result.Add(new MLModel
                {
                    Name = modelName,
                    Description = $"A sentiment analysis model trained using Python and scikit-learn. File: {file}",
                    Language = ModelLanguage.Python
                });
            }

            return Ok(result);
        }

        [HttpGet]
        [Route("Predict")]
        public IActionResult Predict([FromQuery]string modelName, [FromQuery]string language, [FromQuery] string review)
        {
            if(language == "CSharp")
            {
                var predictionService = new PredictionServices();
                var repoRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));
                var model = new MLModel { Name = modelName, Language = ModelLanguage.CSharp };
                var prediction = predictionService.PredictWithMlNet(repoRoot, review, model);
                return Ok(prediction);
            }
            else if(language == "Python")
            {
                var predictionService = new PredictionServices();
                var repoRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));
                var prediction = predictionService.PredictWithOnnx(repoRoot, review);
                return Ok(prediction);
            }
            else
            {
                return BadRequest();
            }
        }


        [HttpPost]
        [Route("Train")]
        public async Task<IActionResult> Train([FromBody] TrainData trainData)
        {
            if (trainData.ModelLanguage == ModelLanguage.CSharp)
            {
                var result = new ModelTrainer().TrainModel(trainData);
                return Ok(result);
            }
            else if (trainData.ModelLanguage == ModelLanguage.Python)
            {
                return Ok();
            }
            else
            {
                return BadRequest();
            }
        }

        [HttpDelete]
        [Route("Delete")]
        public IActionResult Delete([FromQuery] string modelName, [FromQuery] string language)
        {
            var repoRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));

            string filePath = language == "CSharp"
                ? Path.Combine(repoRoot, "models", "sentiment-analysis", "csharp", $"{modelName}.zip")
                : Path.Combine(repoRoot, "models", "sentiment-analysis", "python", $"{modelName}.onnx");

            if (!System.IO.File.Exists(filePath))
                return NotFound($"Model '{modelName}' nije pronađen.");

            System.IO.File.Delete(filePath);
            return Ok();
        }
    }
}

