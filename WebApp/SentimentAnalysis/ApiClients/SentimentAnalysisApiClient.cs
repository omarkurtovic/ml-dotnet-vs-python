using SharedCL.SentimentAnalysis.Models;
using SharedCL.Shared.Enums;
using SharedCL.Shared.Models;
using System.Net.Http.Headers;

namespace WebApp.SentimentAnalysis.ApiClients
{
    public class SentimentAnalysisApiClient(HttpClient httpClient)
    {
        private readonly HttpClient _httpClient = httpClient;

        public async Task<Result<List<SentimentAnalysisModel>>> GetModelsAsync()
        {
            try
            {
                string url = $"SentimentAnalysis/GetModels";
                var request = new HttpRequestMessage(HttpMethod.Get, url);
                var response = await _httpClient.SendAsync(request);
                if (response.IsSuccessStatusCode)
                {
                    var models = await response.Content.ReadFromJsonAsync<List<SentimentAnalysisModel>>();
                    return Result<List<SentimentAnalysisModel>>.Success(models!);
                }
                else
                {
                    return Result<List<SentimentAnalysisModel>>.Failure("");
                }
            }
            catch (Exception ex)
            {
                return Result<List<SentimentAnalysisModel>>.Failure("");
            }
        }

        public async Task<SentimentPrediction> Predict(MLModel model, string review)
        {
            try
            {
                string url = $"SentimentAnalysis/Predict?modelName={Uri.EscapeDataString(model.Name)}&language={Uri.EscapeDataString(model.Language.ToString())}&review={Uri.EscapeDataString(review)}";
                var request = new HttpRequestMessage(HttpMethod.Get, url);
                var response = await _httpClient.SendAsync(request);
                if (response.IsSuccessStatusCode)
                {
                    var prediction = await response.Content.ReadFromJsonAsync<SentimentPrediction>();
                    return prediction!;
                }
                else
                {
                    Console.WriteLine("Failed to get prediction. Status code: " + response.StatusCode);
                    return new SentimentPrediction();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("An error occurred while getting prediction: " + ex.Message);
                return new SentimentPrediction();
            }
        }

        public async Task<Result<SentimentAnalysisModelDto>> TrainModelAsync(TrainData trainDto)
        {
            try
            {
                string url = $"SentimentAnalysis/Train";
                var request = new HttpRequestMessage(HttpMethod.Post, url);
                request.Content = JsonContent.Create(trainDto);
                request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
                var response = await _httpClient.SendAsync(request);
                if (response.IsSuccessStatusCode)
                {
                    var performance = await response.Content.ReadFromJsonAsync<Result<SentimentAnalysisModelDto>>();
                    return performance!;
                }
                else
                {
                    Console.WriteLine("Failed to train model. Status code: " + response.StatusCode);
                    return Result<SentimentAnalysisModelDto>.Failure("Greška prilikom treniranja modela.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("An error occurred while training the model: " + ex.Message);
                return Result<SentimentAnalysisModelDto>.Failure("Greška prilikom treniranja modela.");
            }
        }

        public async Task<Result<bool>> DeleteModelAsync(string modelName, ModelLanguage language)
        {
            try
            {
                var url = $"SentimentAnalysis/Delete?modelName={Uri.EscapeDataString(modelName)}&language={Uri.EscapeDataString(language.ToString())}";
                var response = await _httpClient.DeleteAsync(url);
                return response.IsSuccessStatusCode
                    ? Result<bool>.Success(true)
                    : Result<bool>.Failure("");
            }
            catch (Exception ex)
            {
                Console.WriteLine("An error occurred while deleting the model: " + ex.Message);
                return Result<bool>.Failure("");
            }
        }
    }
}

