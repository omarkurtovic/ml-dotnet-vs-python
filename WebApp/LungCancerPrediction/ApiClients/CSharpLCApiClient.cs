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

        public async Task<Result<List<LCBasic>>> GetModelsBasicAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync("LungCancer/Models/Basic");
                if (response.IsSuccessStatusCode)
                {
                    var models = await response.Content.ReadFromJsonAsync<List<LCBasic>>() ?? new();
                    return Result<List<LCBasic>>.Success(models);
                }
                if (response.StatusCode == System.Net.HttpStatusCode.Unauthorized)
                {
                    return Result<List<LCBasic>>.Failure("Unauthorized access.", FailureReason.Unauthorized);
                }
                Console.WriteLine($"Error fetching models! Status Code: {response.StatusCode}!");
                return Result<List<LCBasic>>.Failure("Failed to fetch models    .");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return Result<List<LCBasic>>.Failure("An error occurred while fetching models.");
            }
        }

        public async Task<Result<LCBasic>> GetModelBasicAsync(int id)
        {
            try
            {
                var response = await _httpClient.GetAsync($"LungCancer/Models/Basic/{id}");
                if (response.IsSuccessStatusCode)
                {
                    var model = await response.Content.ReadFromJsonAsync<LCBasic>() ?? new LCBasic();
                    return Result<LCBasic>.Success(model);
                }
                if (response.StatusCode == System.Net.HttpStatusCode.Unauthorized)
                {
                    return Result<LCBasic>.Failure("Unauthorized access.", FailureReason.Unauthorized);
                }
                Console.WriteLine($"Error fetching models! Status Code: {response.StatusCode}!");
                return Result<LCBasic>.Failure("Failed to fetch models    .");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return Result<LCBasic>.Failure("An error occurred while fetching models.");
            }
        }

        public async Task<Result<LCGridPageDataDto>> SearchModelsAsync(LCGridPageDataDto options)
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

        public async Task<Result<LCInfo>> GetModelInfoAsync(int id)
        {
            try
            {
                var response = await _httpClient.GetAsync($"LungCancer/Models/Info/{id}");
                if (response.IsSuccessStatusCode)
                {
                    var model = await response.Content.ReadFromJsonAsync<LCInfo>();
                    return Result<LCInfo>.Success(model!);
                }
                else
                {
                    var errorDetails = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"API FAILURE: {errorDetails}");
                    return Result<LCInfo>.Failure("");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result<LCInfo>.Failure("");
            }
        }

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

                Console.WriteLine($"Error fetching models! Status Code: {response.StatusCode}!");
                return Result<List<LCDto>>.Failure("Failed to fetch models    .");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return Result<List<LCDto>>.Failure("An error occurred while fetching models.");
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

        public async Task<Result<LCDto>> TrainModelAsync(LCTrainingParamsDto trainingParams)
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
                    var performance = await response.Content.ReadFromJsonAsync<LCDto>();
                    return Result<LCDto>.Success(performance!);
                }
                else
                {
                    var errorDetails = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"API FAILURE: {errorDetails}");
                    return Result<LCDto>.Failure("");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result<LCDto>.Failure("");
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


