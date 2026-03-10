using CSharpModelTrainerApi.SentimentAnalysis.Services;
using Microsoft.AspNetCore.Mvc;
using SharedCL.SentimentAnalysis.Models;
using SharedCL.Shared.Enums;
using SharedCL.Shared.Models;

namespace CSharpModelTrainerApi.SentimentAnalysis.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class SentimentAnalysisController : ControllerBase
    {
        private SentimentAnalysisModelTrainer ModelTrainer { get; set; }
        private SentimentAnalysisPredictionServices SentimentAnalysisPredictionServices { get; set; }
        private SentimentAnalysisRepository SentimentAnalysisRepository { get; set; }

        public SentimentAnalysisController(SentimentAnalysisModelTrainer modelTrainer, 
            SentimentAnalysisRepository sentimentAnalysisRepository,
            SentimentAnalysisPredictionServices sentimentAnalysisPredictionServices)
        {
            ModelTrainer = modelTrainer;
            SentimentAnalysisRepository = sentimentAnalysisRepository;
            SentimentAnalysisPredictionServices = sentimentAnalysisPredictionServices;
        }

        [HttpGet]
        [Route("GetModels")]
        public async Task<IActionResult> GetModels()
        {
            var result = await SentimentAnalysisRepository.GetAll();
            if (!result.IsSuccess)
            {
                return BadRequest();
            }
            else
            {
                return Ok(result.Data);
            }
        }

        [HttpGet]
        [Route("Predict")]
        public async Task<IActionResult> Predict([FromQuery]int id, [FromQuery] string review)
        {
            var modelResult = await SentimentAnalysisRepository.GetById(id);
            if(!modelResult.IsSuccess)
            {
                return BadRequest();
            }

            var model = modelResult.Data;
            if (model == null)
            {
                return NotFound();
            }


            var prediction = SentimentAnalysisPredictionServices.Predict(model, review);
            return Ok(prediction);
        }


        [HttpPost]
        [Route("Train")]
        public IActionResult Train([FromBody] TrainData trainData)
        {
            if (trainData.ModelLanguage == ModelLanguage.CSharp)
            {
                var modelRes = ModelTrainer.TrainModel(trainData);
                if (!modelRes.IsSuccess)
                {
                    return BadRequest();
                }
                return Ok(modelRes.Data);
            }
            else
            {
                return BadRequest();
            }
        }

        [HttpPost]
        [Route("Save")]
        public async Task<IActionResult> Save([FromBody] SentimentAnalysisModel model)
        {
            var saveResult = await SentimentAnalysisRepository.Save(model);
            if (!saveResult.IsSuccess)
                return BadRequest(saveResult);

            return Ok();
        }

        [HttpDelete]
        [Route("Delete")]
        public async Task<IActionResult> Delete([FromQuery] int id)
        {
            var modelResult = await SentimentAnalysisRepository.GetById(id);
            if(!modelResult.IsSuccess)
            {
                return BadRequest();
            }

            var model = modelResult.Data;
            if (model == null)
            {
                return NotFound();
            }

            if (model.Language != ModelLanguage.CSharp)
            {
                return BadRequest();
            }

            var deleteResult = await SentimentAnalysisRepository.Delete(model.Id);
            if(!deleteResult.IsSuccess)
            {
                return BadRequest();
            }

            var repoRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));

            string filePath = Path.Combine(repoRoot, "models", "sentiment-analysis", "csharp", $"{model.Name}.zip");

            if (!System.IO.File.Exists(filePath))
                return Ok();

            System.IO.File.Delete(filePath);
            return Ok();
        }
    }
}

