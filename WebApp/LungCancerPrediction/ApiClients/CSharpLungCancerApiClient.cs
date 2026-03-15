using Microsoft.AspNetCore.Components.Forms;
using SharedCL.LungCancerPrediction.Models;
using SharedCL.Shared.Models;
using System.Net.Http.Headers;

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
                    return Result<List<LungCancerModel>>.Failure("");
                }
            }
            catch (Exception ex)
            {
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
                    Console.WriteLine("Failed to get prediction. Status code: " + response.StatusCode);
                    return Result<LungCancerPredictionModel>.Failure("");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("An error occurred while getting prediction: " + ex.Message);
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
                    return Result<LungCancerModel>.Failure("");
                }
            }
            catch (Exception ex)
            {
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
                var url = $"LungCancer/Delete?id={id}";
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


