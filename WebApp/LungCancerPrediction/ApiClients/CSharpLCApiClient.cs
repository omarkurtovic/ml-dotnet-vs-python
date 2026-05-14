using Microsoft.AspNetCore.Components.Forms;
using SharedCL;
using System.Net.Http.Headers;
using System.Text.Json;

namespace WebApp.LungCancerPrediction.ApiClients
{
    public class CSharpLCApiClient(HttpClient httpClient)
    {
        private readonly HttpClient _httpClient = httpClient;

        public async Task<Result<List<LCDto>>> GetModelsAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync("LungCancer/Models");
                if (response.IsSuccessStatusCode)
                {
                    var models = await response.Content.ReadFromJsonAsync<List<LCDto>>() ?? new();
                    return Result<List<LCDto>>.Success(models);
                }
                if (response.StatusCode == System.Net.HttpStatusCode.Unauthorized)
                {
                    return Result<List<LCDto>>.Failure("Unauthorized access.", FailureReason.Unauthorized);
                }
                Console.WriteLine($"Error fetching model names! Status Code: {response.StatusCode}!");
                return Result<List<LCDto>>.Failure("Failed to fetch model names.");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return Result<List<LCDto>>.Failure("An error occurred while fetching model names.");
            }
        }

        public async Task<Result<LCDto>> GetModelAsync(int id)
        {
            try
            {
                var response = await _httpClient.GetAsync($"LungCancer/Models/{id}");
                if (response.IsSuccessStatusCode)
                {
                    var model = await response.Content.ReadFromJsonAsync<LCDto>() ?? new LCDto();
                    return Result<LCDto>.Success(model);
                }
                if (response.StatusCode == System.Net.HttpStatusCode.Unauthorized)
                {
                    return Result<LCDto>.Failure("Unauthorized access.", FailureReason.Unauthorized);
                }

                Console.WriteLine($"Error fetching models! Status Code: {response.StatusCode}!");
                return Result<LCDto>.Failure("Failed to fetch models    .");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return Result<LCDto>.Failure("An error occurred while fetching models.");
            }
        }

        public async Task<Result<List<LCBasicDto>>> GetModelsBasicAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync("LungCancer/Models/Basic");
                if (response.IsSuccessStatusCode)
                {
                    var models = await response.Content.ReadFromJsonAsync<List<LCBasicDto>>() ?? new();
                    return Result<List<LCBasicDto>>.Success(models);
                }
                if (response.StatusCode == System.Net.HttpStatusCode.Unauthorized)
                {
                    return Result<List<LCBasicDto>>.Failure("Unauthorized access.", FailureReason.Unauthorized);
                }
                Console.WriteLine($"Error fetching models! Status Code: {response.StatusCode}!");
                return Result<List<LCBasicDto>>.Failure("Failed to fetch models    .");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return Result<List<LCBasicDto>>.Failure("An error occurred while fetching models.");
            }
        }

        public async Task<Result<LCBasicDto>> GetModelBasicAsync(int id)
        {
            try
            {
                var response = await _httpClient.GetAsync($"LungCancer/Models/Basic/{id}");
                if (response.IsSuccessStatusCode)
                {
                    var model = await response.Content.ReadFromJsonAsync<LCBasicDto>() ?? new LCBasicDto();
                    return Result<LCBasicDto>.Success(model);
                }
                if (response.StatusCode == System.Net.HttpStatusCode.Unauthorized)
                {
                    return Result<LCBasicDto>.Failure("Unauthorized access.", FailureReason.Unauthorized);
                }
                Console.WriteLine($"Error fetching models! Status Code: {response.StatusCode}!");
                return Result<LCBasicDto>.Failure("Failed to fetch models    .");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return Result<LCBasicDto>.Failure("An error occurred while fetching models.");
            }
        }

        public async Task<Result<LCGridPageDataDto>> GetModelsSearchAsync(LCGridOptionsDto options)
        {
            try
            {
                var response = await _httpClient.PostAsJsonAsync("LungCancer/Models/Search", options);
                if (response.IsSuccessStatusCode)
                {
                    var models = await response.Content.ReadFromJsonAsync<LCGridPageDataDto>() ?? new();
                    return Result<LCGridPageDataDto>.Success(models);
                }
                if (response.StatusCode == System.Net.HttpStatusCode.Unauthorized)
                {
                    return Result<LCGridPageDataDto>.Failure("Unauthorized access.", FailureReason.Unauthorized);
                }

                Console.WriteLine($"Error fetching models! Status Code: {response.StatusCode}!");
                return Result<LCGridPageDataDto>.Failure("Failed to fetch models    .");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return Result<LCGridPageDataDto>.Failure("An error occurred while fetching models.");
            }
        }

        public async Task<Result<LCInfoDto>> GetModelInfoAsync(int id)
        {
            try
            {
                var response = await _httpClient.GetAsync($"LungCancer/Models/Info/{id}");
                if (response.IsSuccessStatusCode)
                {
                    var model = await response.Content.ReadFromJsonAsync<LCInfoDto>();
                    return Result<LCInfoDto>.Success(model!);
                }
                else
                {
                    var errorDetails = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"API FAILURE: {errorDetails}");
                    return Result<LCInfoDto>.Failure("");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result<LCInfoDto>.Failure("");
            }
        }

        public async Task<Result<LCPredictionDto>> PredictAsync(int id, IBrowserFile file)
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
                    var prediction = await response.Content.ReadFromJsonAsync<LCPredictionDto>();
                    return Result<LCPredictionDto>.Success(prediction!);
                }
                else
                {
                    var errorDetails = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"API FAILURE: {errorDetails}");
                    return Result<LCPredictionDto>.Failure("");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result<LCPredictionDto>.Failure("");
            }
        }

        public async Task<Result<int>> TrainModelAsync(LCTrainingParamsDto trainingParams)
        {
            try
            {
                string url = $"LungCancer/Train";
                var request = new HttpRequestMessage(HttpMethod.Post, url)
                {
                    Content = JsonContent.Create(trainingParams)
                };
                request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
                var response = await _httpClient.SendAsync(request);
                if (response.IsSuccessStatusCode)
                {
                    var performance = await response.Content.ReadFromJsonAsync<int>();
                    return Result<int>.Success(performance!);
                }
                else
                {
                    var errorDetails = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"API FAILURE: {errorDetails}");
                    return Result<int>.Failure("");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result<int>.Failure("");
            }
        }

        public async Task<Result> SaveModelAsync(LCDto model)
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


