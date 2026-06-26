using CSharpModelTrainerApi.Enums;
using CSharpModelTrainerApi.LungCancerPrediction.Services;
using CSharpModelTrainerApi.Services;
using Microsoft.AspNetCore.Mvc;
using SharedCL;

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
        [Route("Models/{id}")]
        public async Task<IActionResult> GetModel([FromRoute] int id)
        {
            var modelResult = await LungCancerModelRepository.GetModel(id);
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

        [HttpGet]
        [Route("Models/Basic")]
        public async Task<IActionResult> GetModelsBasic()
        {
            var result = await LungCancerModelRepository.GetModelsBasic();
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
        [Route("Models/Basic/{id}")]
        public async Task<IActionResult> GetModelBasic([FromRoute] int id)
        {
            var modelResult = await LungCancerModelRepository.GetModelBasic(id);
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
        [Route("Models/Search")]
        public async Task<IActionResult> GetModels([FromBody] LCGridOptionsDto options)
        {
            var result = await LungCancerModelRepository.GetModelsSearch(options);
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
        [Route("Models/Info/{id}")]
        public async Task<IActionResult> GetModelInfo([FromRoute] int id)
        {
            var modelResult = await LungCancerModelRepository.GetModelInfo(id);
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
            var modelResult = await LungCancerModelRepository.GetModel(id);
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
            var saveResult = await LungCancerModelRepository.Save(modelRes.Data!);
            if (!saveResult.IsSuccess)
            {
                return BadRequest(saveResult.Message);
            }

            return Ok(saveResult.Data);
        }

        [HttpPost]
        [Route("Save")]
        public async Task<IActionResult> Save([FromBody] LCDto model)
        {
            var saveResult = await LungCancerModelRepository.Save(model);
            if (!saveResult.IsSuccess)
                return BadRequest(saveResult.Message);
            return Ok(saveResult.Data);
        }

        [HttpDelete]
        [Route("Delete")]
        public async Task<IActionResult> Delete([FromQuery] int id)
        {
            var modelResult = await LungCancerModelRepository.GetModelBasic(id);
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

        [HttpPost]
        [Route("UpdateModelName")]
        public async Task<IActionResult> UpdateModelName([FromQuery] int id, [FromBody] string newName)
        {
            var updateResult = await LungCancerModelRepository.UpdateNameAsync(id, newName);
            if (!updateResult.IsSuccess)
            {
                return BadRequest(updateResult.Message);
            }
            return Ok();
        }
    }
}
