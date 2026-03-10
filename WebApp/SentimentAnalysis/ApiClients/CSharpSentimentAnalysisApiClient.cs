using SharedCL.SentimentAnalysis.Models;
using SharedCL.Shared.Enums;
using SharedCL.Shared.Models;
using System.Net.Http.Headers;

namespace WebApp.SentimentAnalysis.ApiClients
{
    public class CSharpSentimentAnalysisApiClient(HttpClient httpClient)
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

        public async Task<SentimentPrediction> Predict(int id, string review)
        {
            try
            {
                string url = $"SentimentAnalysis/Predict?id={id}&review={Uri.EscapeDataString(review)}";
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

        public async Task<Result<SentimentAnalysisModel>> TrainModelAsync(SentimentAnalysisTrainingParams trainDto)
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
                    var performance = await response.Content.ReadFromJsonAsync<SentimentAnalysisModel>();
                    return Result<SentimentAnalysisModel>.Success(performance!);
                }
                else
                {
                    return Result<SentimentAnalysisModel>.Failure("");
                }
            }
            catch (Exception ex)
            {
                return Result<SentimentAnalysisModel>.Failure("");
            }
        }
        public async Task<Result> SaveModelAsync(SentimentAnalysisModel model)
        {
            try
            {
                string url = $"SentimentAnalysis/Save";
                var request = new HttpRequestMessage(HttpMethod.Post, url);
                request.Content = JsonContent.Create(model);
                request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
                var response = await _httpClient.SendAsync(request);
                if (response.IsSuccessStatusCode)
                {
                    return Result.Success();
                }
                else
                {
                    return Result.Failure("");
                }
            }
            catch (Exception ex)
            {
                return Result.Failure("");
            }
        }

        public async Task<Result<bool>> DeleteModelAsync(int id)
        {
            try
            {
                var url = $"SentimentAnalysis/Delete?id={id}";
                var response = await _httpClient.DeleteAsync(url);
                return response.IsSuccessStatusCode
                    ? Result<bool>.Success(true)
                    : Result<bool>.Failure("");
            }
            catch (Exception ex)
            {
                return Result<bool>.Failure("");
            }
        }
    }
}

