using SharedCL.SentimentAnalysis.Dtos;
using SharedCL.SentimentAnalysis.Models;
using SharedCL.Shared.Enums;
using SharedCL.Shared.Models;
using System.Net.Http.Headers;

namespace WebApp.SentimentAnalysis.ApiClients
{
    public class SentimentAnalysisApiClient(HttpClient httpClient)
    {
        private readonly HttpClient _httpClient = httpClient;

        public async Task<Result<List<MLModel>>> GetModelsAsync()
        {
            try
            {
                string url = $"SentimentAnalysis/GetModels";
                var request = new HttpRequestMessage(HttpMethod.Get, url);
                var response = await _httpClient.SendAsync(request);
                if (response.IsSuccessStatusCode)
                {
                    var models = await response.Content.ReadFromJsonAsync<List<MLModel>>();
                    return Result<List<MLModel>>.Success(models!);
                }
                else
                {
                    Console.WriteLine("Failed to fetch models. Status code: " + response.StatusCode);
                    return Result<List<MLModel>>.Failure("");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("An error occurred while fetching models: " + ex.Message);
                return Result<List<MLModel>>.Failure("");
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

        public async Task<Result<ModelPerformance>> TrainModelAsync(TrainData trainDto)
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
                    var performance = await response.Content.ReadFromJsonAsync<ModelPerformance>();
                    return Result<ModelPerformance>.Success(performance!);
                }
                else
                {
                    Console.WriteLine("Failed to train model. Status code: " + response.StatusCode);
                    return Result<ModelPerformance>.Failure("");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("An error occurred while training the model: " + ex.Message);
                return Result<ModelPerformance>.Failure("");
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

