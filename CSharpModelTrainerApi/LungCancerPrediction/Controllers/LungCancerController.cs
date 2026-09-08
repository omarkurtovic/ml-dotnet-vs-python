using CSharpModelTrainerApi.Enums;
using CSharpModelTrainerApi.LungCancerPrediction.Models;
using CSharpModelTrainerApi.LungCancerPrediction.Services;
using CSharpModelTrainerApi.LungCancerPrediction.Workers;
using CSharpModelTrainerApi.Mappers;
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


        [HttpGet]
        [Route("Models/Comparison")]
        public async Task<IActionResult> GetModels()
        {
            var result = await LungCancerModelRepository.GetModelsForComparison();
            if (!result.IsSuccess)
            {
                return BadRequest();
            }
            else
            {
                return Ok(result.Data!.ToComparisonDto());
            }
        }

        [HttpGet]
        [Route("Models/Overview")]
        public async Task<IActionResult> GetModelsBasic()
        {
            var result = await LungCancerModelRepository.GetModelsForOverview();
            if (!result.IsSuccess)
            {
                return BadRequest();
            }
            else
            {
                return Ok(result.Data!.ToOverviewDto());
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

            if(model.Language != ModelLanguage.Python || model.TrainingStatus != LCTrainingStatus.Training)
            {
                return Ok(MapToDto(model));
            }

            var trainingProgress = await PythonLCApi.GetTrainingProgressAsync(id);
            if(trainingProgress == null || trainingProgress.Count == 0)
            {
                return Ok(MapToDto(model));
            }

            model.TrainingTimeInSeconds = trainingProgress.Last().TrainingTimeInSeconds;
            model.EpochData ??= [];
            if (trainingProgress.Last().TrainingStatus != TrainingStatusDto.Training)
            {
                model.TrainingStatus = (LCTrainingStatus)trainingProgress.Last().TrainingStatus;
            }

            for (int i = model.EpochData.Count; i < trainingProgress.Count; ++i)
            {
                var progress = trainingProgress[i];
                model.EpochData.Add(new LCEpochData
                {
                    LCModelId = id,
                    Epoch = progress.Epoch,
                    TrainingLoss = progress.TrainingLoss,
                    TrainingAccuracy = progress.TrainingAccuracy,
                    ValidationLoss = progress.ValidationLoss,
                    ValidationAccuracy = progress.ValidationAccuracy,
                    BenignPrecision = progress.BenignPrecision,
                    BenignRecall = progress.BenignRecall,
                    BenignF1Score = progress.BenignF1Score,
                    MalignantPrecision = progress.MalignantPrecision,
                    MalignantRecall = progress.MalignantRecall,
                    MalignantF1Score = progress.MalignantF1Score,
                    NormalPrecision = progress.NormalPrecision,
                    NormalRecall = progress.NormalRecall,
                    NormalF1Score = progress.NormalF1Score,
                    MacroPrecision = progress.MacroPrecision,
                    MacroRecall = progress.MacroRecall,
                    MacroF1Score = progress.MacroF1Score,
                    WeightedPrecision = progress.WeightedPrecision,
                    WeightedRecall = progress.WeightedRecall,
                    WeightedF1Score = progress.WeightedF1Score,
                    ValidationScores = [.. progress.ValidationScores?.Select(p => new LCValidationScore
                    {
                        BenignProbability = p.BenignProbability,
                        MalignantProbability = p.MalignantProbability,
                        NormalProbability = p.NormalProbability,
                        TrueLabel = p.TrueLabel
                    }) ?? []]
                });
            }

            var saveResult = await LungCancerModelRepository.SaveChangesAsync();
            if (!saveResult.IsSuccess)
            {
                return BadRequest("Greška prilikom spremanja modela");
            }

            return Ok(MapToDto(model));
        }

        private static LCInfoDto MapToDto(LCModel model)
        {
            var lastEpoch = model.EpochData?.LastOrDefault();
            var roc = ROCService.Calculate(lastEpoch?.ValidationScores ?? []);
            return model.ToInfoDto(roc);
        }

        [HttpPost]
        [Route("Predict")]
        public async Task<IActionResult> Predict([FromQuery] int id, [FromForm] IFormFile file)
        {
            var modelResult = await LungCancerModelRepository.GetModelDto(id);
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
                    await LungCancerModelRepository.UpdateStatusAsync(saveResult.Data, LCTrainingStatus.Failed);
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
