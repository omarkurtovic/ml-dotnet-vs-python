using Microsoft.AspNetCore.Components.Forms;
using SharedCL.LungCancerPrediction.Models;
using SharedCL.Shared.Models;
using System.Net.Http.Headers;
using System.Text.Json;

namespace WebApp.LungCancerPrediction.ApiClients
{
    public class CSharpLungCancerApiClient(HttpClient httpClient)
    {
        private readonly HttpClient _httpClient = httpClient;

        public async Task<Result<List<LungCancerModel>>> GetModelsAsync()
        {
            try
            {
                string url = $"LungCancer/GetModels";
                var request = new HttpRequestMessage(HttpMethod.Get, url);
                var response = await _httpClient.SendAsync(request);
                if (response.IsSuccessStatusCode)
                {
                    var models = await response.Content.ReadFromJsonAsync<List<LungCancerModel>>();
                    return Result<List<LungCancerModel>>.Success(models!);
                }
                else
                {
                    var errorDetails = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"API FAILURE: {errorDetails}");
                    return Result<List<LungCancerModel>>.Failure("");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result<List<LungCancerModel>>.Failure("");
            }
        }

        public async Task<Result<LungCancerPredictionModel>> PredictAsync(int id, IBrowserFile file)
        {
            try
            {
                string url = $"LungCancer/Predict?id={id}";
                var request = new HttpRequestMessage(HttpMethod.Post, url);
                request.Content = new MultipartFormDataContent
                {
                    { new StreamContent(file.OpenReadStream()), "file", file.Name }
                };
                var response = await _httpClient.SendAsync(request);
                if (response.IsSuccessStatusCode)
                {
                    var prediction = await response.Content.ReadFromJsonAsync<LungCancerPredictionModel>();
                    return Result<LungCancerPredictionModel>.Success(prediction!);
                }
                else
                {
                    var errorDetails = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"API FAILURE: {errorDetails}");
                    return Result<LungCancerPredictionModel>.Failure("");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result<LungCancerPredictionModel>.Failure("");
            }
        }

        public async Task<Result<LungCancerModel>> TrainModelAsync(LungCancerTrainingParams trainingParams)
        {
            try
            {
                string url = $"LungCancer/Train";
                var request = new HttpRequestMessage(HttpMethod.Post, url);
                request.Content = JsonContent.Create(trainingParams);
                request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
                var response = await _httpClient.SendAsync(request);
                if (response.IsSuccessStatusCode)
                {
                    var performance = await response.Content.ReadFromJsonAsync<LungCancerModel>();
                    return Result<LungCancerModel>.Success(performance!);
                }
                else
                {
                    var errorDetails = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"API FAILURE: {errorDetails}");
                    return Result<LungCancerModel>.Failure("");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result<LungCancerModel>.Failure("");
            }
        }
        public async Task<Result> SaveModelAsync(LungCancerModel model)
        {
            try
            {
                string url = $"LungCancer/Save";
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
                    var errorDetails = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"API FAILURE: {errorDetails}");
                    return Result.Failure("");

                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result.Failure("");
            }
        }

        public async Task<Result<bool>> DeleteModelAsync(int id)
        {
            try
            {
                var url = $"LungCancer/Delete?id={id}";
                var response = await _httpClient.DeleteAsync(url);
                if (response.IsSuccessStatusCode)
                {
                    return Result<bool>.Success(true);
                }
                else
                {
                    var errorDetails = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"API FAILURE: {errorDetails}");
                    return Result<bool>.Failure("");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result<bool>.Failure("");
            }
        }
    }
}


