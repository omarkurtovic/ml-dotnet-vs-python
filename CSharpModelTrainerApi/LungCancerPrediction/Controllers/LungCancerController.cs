using CSharpModelTrainerApi.LungCancerPrediction.Services;
using CSharpModelTrainerApi.SentimentAnalysis.Services;
using CSharpModelTrainerApi.Shared;
using Microsoft.AspNetCore.Mvc;
using SharedCL.LungCancerPrediction.Dtos;
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
        private LungCancerPredictionService LungCancerPredictionService { get; set; }
        private LungCancerModelRepository LungCancerModelRepository { get; set; }
        private PathResolver PathResolver { get; set; }
        private HardwareInfoService HardwareInfoService { get; set; }
        public LungCancerController(LungCancerModelTrainer modelTrainer,
            LungCancerModelRepository lungCancerModelRepository,
            LungCancerPredictionService lungCancerPredictionService,
            PathResolver pathResolver,
            HardwareInfoService hardwareInfoService)
        {
            ModelTrainer = modelTrainer;
            LungCancerModelRepository = lungCancerModelRepository;
            LungCancerPredictionService = lungCancerPredictionService;
            PathResolver = pathResolver;
            HardwareInfoService = hardwareInfoService;
        }

        [HttpPost]
        [Route("search")]
        public async Task<IActionResult> GetModels([FromBody] LCModelsGridOptionsDto options)
        {
            var result = await LungCancerModelRepository.GetModelsPageData(options);
            if (!result.IsSuccess)
            {
                return BadRequest();
            }
            else
            {
                return Ok(result.Data);
            }
        }

        [HttpPost]
        [Route("Predict")]
        public async Task<IActionResult> Predict([FromQuery] int id, [FromForm] IFormFile file)
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


            var prediction = await LungCancerPredictionService.Predict(model, file);
            return Ok(prediction);
        }


        [HttpPost]
        [Route("Train")]
        public IActionResult Train([FromBody] LungCancerTrainingParams trainParams)
        {
            var modelRes = ModelTrainer.TrainModel(trainParams, HardwareInfoService);
            if (!modelRes.IsSuccess)
            {
                return BadRequest(modelRes.Message);
            }
            return Ok(modelRes.Data);
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

            var deleteResult = await LungCancerModelRepository.Delete(model.Id);
            if (!deleteResult.IsSuccess)
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
