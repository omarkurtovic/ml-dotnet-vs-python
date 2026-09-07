using CSharpModelTrainerApi.Enums;
using CSharpModelTrainerApi.LungCancerPrediction.Services;
using CSharpModelTrainerApi.LungCancerPrediction.Workers;
using CSharpModelTrainerApi.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using SharedCL;
using System.IO;

namespace CSharpModelTrainerApi.LungCancerPrediction.Controllers
{

    [ApiController]
    [Route("[controller]")]
    public class LungCancerController(
        LCRepository lungCancerModelRepository,
        LCPredictionService lungCancerPredictionService,
        PathResolver pathResolver,
        HardwareInfoService hardwareInfoService,
        TrainingQueue trainingQueue,
        PythonLCApiClient pythonLCApiClient,
        ROCService rocService) : ControllerBase
    {
        private LCPredictionService LungCancerPredictionService { get; set; } = lungCancerPredictionService;
        private LCRepository LungCancerModelRepository { get; set; } = lungCancerModelRepository;
        private PathResolver PathResolver { get; set; } = pathResolver;
        private HardwareInfoService HardwareInfoService { get; set; } = hardwareInfoService;
        private TrainingQueue _trainingQueue { get; set; } = trainingQueue;
        private PythonLCApiClient PythonLCApi { get; set; } = pythonLCApiClient;
        private readonly ROCService _rocService = rocService;


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

            if (model.Language == ModelLanguageDto.Python && model.ModelStatusDto == TrainingStatusDto.Training)
            {
                try
                {
                    var trainingProgress = await PythonLCApi.GetTrainingInfoAsync(id);
                    if (trainingProgress != null && trainingProgress.Count > 0)
                    {
                        await LungCancerModelRepository.UpdateTrainingTimeAsync(id, trainingProgress.Last);

                        var persistedModelResult = await LungCancerModelRepository.GetModel(id);
                        var persistedEpochs = persistedModelResult.Data?.EpochData?.Count ?? 0;
                        if (trainingProgress.CurrentEpoch > persistedEpochs)
                        {
                            await LungCancerModelRepository.AddEpochData(id, new LCEpochDataDto
                            {
                                Epoch = trainingProgress.CurrentEpoch - 1,
                                TrainingLoss = trainingProgress.TrainingLoss,
                                TrainingAccuracy = trainingProgress.TrainingAccuracy,
                                ValidationLoss = trainingProgress.ValidationLoss,
                                ValidationAccuracy = trainingProgress.ValidationAccuracy,
                                BenignPrecision = trainingProgress.BenignPrecision,
                                BenignRecall = trainingProgress.BenignRecall,
                                BenignF1Score = trainingProgress.BenignF1Score,
                                MalignantPrecision = trainingProgress.MalignantPrecision,
                                MalignantRecall = trainingProgress.MalignantRecall,
                                MalignantF1Score = trainingProgress.MalignantF1Score,
                                NormalPrecision = trainingProgress.NormalPrecision,
                                NormalRecall = trainingProgress.NormalRecall,
                                NormalF1Score = trainingProgress.NormalF1Score,
                                MacroPrecision = trainingProgress.MacroPrecision,
                                MacroRecall = trainingProgress.MacroRecall,
                                MacroF1Score = trainingProgress.MacroF1Score,
                                WeightedPrecision = trainingProgress.WeightedPrecision,
                                WeightedRecall = trainingProgress.WeightedRecall,
                                WeightedF1Score = trainingProgress.WeightedF1Score,
                                LCPredictions = trainingProgress.LCPredictions
                            });
                        }

                        if (trainingProgress.ModelStatusDto != TrainingStatusDto.Training)
                        {
                            await LungCancerModelRepository.UpdateStatusAsync(
                                id, (TrainingStatus)trainingProgress.ModelStatusDto);
                        }
                        return Ok(trainingProgress);
                    }
                }
                catch (HttpRequestException)
                {
                }
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
            if (string.IsNullOrEmpty(trainParams.Name))
            {
                return BadRequest("Naziv modela ne smije biti prazan");
            }
            if (trainParams.Epochs < 1 || trainParams.Epochs > 100)
            {
                return BadRequest("Broj epoha mora biti između 1 i 100");
            }

            var modelDB = new LCDto
            {
                Name = trainParams.Name,
                Language = (ModelLanguageDto)trainParams.Language,
                EpochData = [],
                HardwareInfo = hardwareInfoService.GetHardwareInfo(),
                TotalEpochs = trainParams.Epochs,
                ModelStatusDto = TrainingStatusDto.Training
            };

            var saveResult = await LungCancerModelRepository.Save(modelDB);
            if (!saveResult.IsSuccess)
            {
                return BadRequest("Greška prilikom spremanja modela");
            }

            if (trainParams.Language == ModelLanguageDto.CSharp)
            {
                await _trainingQueue.EnqueueAsync(saveResult.Data, trainParams);
            }
            else
            {
                try
                {
                    await PythonLCApi.StartTrainingAsync(saveResult.Data, trainParams);
                }
                catch
                {
                    await LungCancerModelRepository.UpdateStatusAsync(saveResult.Data, TrainingStatus.Failed);
                    return BadRequest("Greška prilikom pokretanja Python treniranja");
                }
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

            var modelPath = PathResolver.GetModelPath(model);

            var deleteResult = await LungCancerModelRepository.Delete(model.Id, () =>
            {
                if (System.IO.File.Exists(modelPath))
                {
                    System.IO.File.Delete(modelPath);
                }
            });

            if (!deleteResult.IsSuccess)
            {
                return BadRequest(deleteResult.Message);
            }

            return Ok();
        }

        [HttpPost]
        [Route("UpdateModelName")]
        public async Task<IActionResult> UpdateModelName([FromQuery] int id, [FromBody] string newName)
        {
            if (string.IsNullOrWhiteSpace(newName))
            {
                return BadRequest("New name cannot be empty.");
            }

            var modelResult = await LungCancerModelRepository.GetModelBasic(id);
            if (!modelResult.IsSuccess)
            {
                return NotFound();
            }
            var model = modelResult.Data;


            if (model!.Name == newName)
            {
                return BadRequest("New name is the same as the current name.");
            }

            var modelPath = PathResolver.GetModelPath(model);

            if (!System.IO.File.Exists(modelPath))
            {
                return NotFound();
            }

            var originalName = model.Name;
            var updateResult = await LungCancerModelRepository.UpdateNameAsync(id, newName);
            if (!updateResult.IsSuccess)
            {
                return BadRequest(updateResult.Message);
            }

            model.Name = newName;
            var newPath = PathResolver.GetModelPath(model);

            try
            {
                System.IO.File.Move(modelPath, newPath);
            }
            catch (IOException)
            {
                await LungCancerModelRepository.UpdateNameAsync(id, originalName);
                return StatusCode(500, "Greška prilikom premještanja fajla modela; naziv je vraćen na prethodni.");
            }

            return Ok();
        }
    }
}
