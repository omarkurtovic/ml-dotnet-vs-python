using CSharpModelTrainerApi.SentimentAnalysis.Services;
using CSharpModelTrainerApi.Shared;
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
        private PathResolver PathResolver { get; set; }

        public SentimentAnalysisController(SentimentAnalysisModelTrainer modelTrainer, 
            SentimentAnalysisRepository sentimentAnalysisRepository,
            SentimentAnalysisPredictionServices sentimentAnalysisPredictionServices,
            PathResolver pathResolver)
        {
            ModelTrainer = modelTrainer;
            SentimentAnalysisRepository = sentimentAnalysisRepository;
            SentimentAnalysisPredictionServices = sentimentAnalysisPredictionServices;
            PathResolver = pathResolver;
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


            var prediction = await SentimentAnalysisPredictionServices.Predict(model, review);
            return Ok(prediction);
        }


        [HttpPost]
        [Route("Train")]
        public async Task<IActionResult> Train([FromBody] SentimentAnalysisTrainingParams trainData)
        {
            if (trainData.ModelLanguage == ModelLanguage.CSharp)
            {
                var modelRes = await ModelTrainer.TrainModel(trainData);
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

            var deleteResult = await SentimentAnalysisRepository.Delete(model.Id);
            if(!deleteResult.IsSuccess)
            {
                return BadRequest();
            }

            var modelPath = PathResolver.GetModelPath(model);
            if (!System.IO.File.Exists(modelPath))
                return Ok();

            System.IO.File.Delete(modelPath);
            return Ok();
        }
    }
}

