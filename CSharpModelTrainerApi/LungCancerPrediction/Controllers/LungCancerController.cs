using CSharpModelTrainerApi.LungCancerPrediction.Models;
using CSharpModelTrainerApi.LungCancerPrediction.Services;
using CSharpModelTrainerApi.SentimentAnalysis.Services;
using Microsoft.AspNetCore.Mvc;
using SharedCL.LungCancerPrediction.Models;
using SharedCL.SentimentAnalysis.Models;
using SharedCL.Shared.Enums;

namespace CSharpModelTrainerApi.LungCancerPrediction.Controllers
{

    [ApiController]
    [Route("[controller]")]
    public class LungCancerController : ControllerBase
    {
        private LungCancerModelTrainer ModelTrainer { get; set; }
        private LungCancerPredictionServices LungCancerPredictionServices { get; set; }
        private LungCancerModelRepository LungCancerModelRepository { get; set; }
        public LungCancerController(LungCancerModelTrainer modelTrainer,
            LungCancerModelRepository lungCancerModelRepository,
            LungCancerPredictionServices lungCancerPredictionServices)
        {
            ModelTrainer = modelTrainer;
            LungCancerModelRepository = lungCancerModelRepository;
            LungCancerPredictionServices = lungCancerPredictionServices ;
        }

        [HttpGet]
        [Route("GetModels")]
        public async Task<IActionResult> GetModels()
        {
            var result = await LungCancerModelRepository.GetAll();
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
        public async Task<IActionResult> Predict([FromQuery] int id, [FromQuery] string review)
        {
            var modelResult = await LungCancerModelRepository.GetById(id);
            if (!modelResult.IsSuccess)
            {
                return BadRequest();
            }

            var model = modelResult.Data;
            if (model == null)
            {
                return NotFound();
            }


            var prediction = LungCancerPredictionServices.Predict(model, review);
            return Ok(prediction);
        }


        [HttpPost]
        [Route("Train")]
        public IActionResult Train([FromBody] LungCancerTrainingParams trainParams)
        {
            if (trainParams.ModelLanguage == ModelLanguage.CSharp)
            {
                var modelRes = ModelTrainer.TrainModel(trainParams);
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
        public async Task<IActionResult> Save([FromBody] LungCancerModel model)
        {
            var saveResult = await LungCancerModelRepository.Save(model);
            if (!saveResult.IsSuccess)
                return BadRequest(saveResult);

            return Ok();
        }

        [HttpDelete]
        [Route("Delete")]
        public async Task<IActionResult> Delete([FromQuery] int id)
        {
            var modelResult = await LungCancerModelRepository.GetById(id);
            if (!modelResult.IsSuccess)
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

            var deleteResult = await LungCancerModelRepository.Delete(model.Id);
            if (!deleteResult.IsSuccess)
            {
                return BadRequest();
            }

            var repoRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));

            string filePath = Path.Combine(repoRoot, "models", "lung-cancer", "csharp", $"{model.Name}.weights");

            if (!System.IO.File.Exists(filePath))
                return Ok();

            System.IO.File.Delete(filePath);
            return Ok();
        }
    }
}
