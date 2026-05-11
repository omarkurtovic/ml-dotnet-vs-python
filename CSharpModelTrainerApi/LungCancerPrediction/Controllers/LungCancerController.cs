using CSharpModelTrainerApi.Enums;
using CSharpModelTrainerApi.LungCancerPrediction.Services;
using CSharpModelTrainerApi.SentimentAnalysis.Services;
using CSharpModelTrainerApi.Services;
using Microsoft.AspNetCore.Mvc;
using SharedCL;
using SharedCL.LungCancerPrediction.Dtos;

namespace CSharpModelTrainerApi.LungCancerPrediction.Controllers
{

    [ApiController]
    [Route("[controller]")]
    public class LungCancerController(LCTrainer modelTrainer,
        LCRepository lungCancerModelRepository,
        LCPredictionService lungCancerPredictionService,
        PathResolver pathResolver,
        HardwareInfoService hardwareInfoService) : ControllerBase
    {
        private LCTrainer ModelTrainer { get; set; } = modelTrainer;
        private LCPredictionService LungCancerPredictionService { get; set; } = lungCancerPredictionService;
        private LCRepository LungCancerModelRepository { get; set; } = lungCancerModelRepository;
        private PathResolver PathResolver { get; set; } = pathResolver;
        private HardwareInfoService HardwareInfoService { get; set; } = hardwareInfoService;

        [HttpPost]
        [Route("Search")]
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

        [HttpGet]
        [Route("Models")]
        public async Task<IActionResult> GetModels()
        {
            var result = await LungCancerModelRepository.GetModels();
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
        [Route("GetDetailsById")]
        public async Task<IActionResult> GetDetailsById([FromQuery] int id)
        {
            var modelResult = await LungCancerModelRepository.GetDetailsById(id);
            if (!modelResult.IsSuccess)
            {
                return BadRequest();
            }
            var model = modelResult.Data;
            if (model == null)
            {
                return NotFound();
            }
            return Ok(model);
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
        public async Task<IActionResult> Train([FromBody] LCTrainingParamsDto trainParams)
        {
            var modelRes = ModelTrainer.TrainModel(trainParams, HardwareInfoService);
            if (!modelRes.IsSuccess)
            {
                return BadRequest(modelRes.Message);
            }
            await LungCancerModelRepository.Save(modelRes.Data!);
            return Ok(LungCancerModelRepository.GetById(modelRes.Data!.Id).Result.Data);
        }

        [HttpPost]
        [Route("Save")]
        public async Task<IActionResult> Save([FromBody] LCWithEpochs model)
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
            var modelResult = await LungCancerModelRepository.GetDetailsById(id);
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
