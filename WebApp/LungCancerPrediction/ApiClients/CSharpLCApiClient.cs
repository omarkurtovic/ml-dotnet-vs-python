using Microsoft.AspNetCore.Components.Forms;
using SharedCL;
using System.Net.Http.Headers;
using System.Text.Json;

namespace WebApp.LungCancerPrediction.ApiClients
{
    public class CSharpLCApiClient(HttpClient httpClient)
    {
        private readonly HttpClient _httpClient = httpClient;

        public async Task<Result<List<LCModelComparisonDto>>> GetModelsForComparisonAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync("LungCancer/Models/Comparison");
                if (response.IsSuccessStatusCode)
                {
                    var models = await response.Content.ReadFromJsonAsync<List<LCModelComparisonDto>>() ?? new();
                    return Result<List<LCModelComparisonDto>>.Success(models);
                }
                if (response.StatusCode == System.Net.HttpStatusCode.Unauthorized)
                {
                    return Result<List<LCModelComparisonDto>>.Failure(Loc.T("LCErrors_UnauthorizedAccess"), FailureReason.Unauthorized);
                }
                Console.WriteLine($"Error fetching model names! Status Code: {response.StatusCode}!");
                return Result<List<LCModelComparisonDto>>.Failure(Loc.T("LCErrors_ErrorFetchingData"));
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return Result<List<LCModelComparisonDto>>.Failure(Loc.T("LCErrors_ErrorFetchingData"));
            }
        }

        public async Task<Result<List<LCModelInferenceDto>>> GetModelsForInference()
        {
            try
            {
                var response = await _httpClient.GetAsync("LungCancer/Models/Inference");
                if (response.IsSuccessStatusCode)
                {
                    var models = await response.Content.ReadFromJsonAsync<List<LCModelInferenceDto>>() ?? [];
                    return Result<List<LCModelInferenceDto>>.Success(models);
                }
                if (response.StatusCode == System.Net.HttpStatusCode.Unauthorized)
                {
                    return Result<List<LCModelInferenceDto>>.Failure(Loc.T("LCErrors_UnauthorizedAccess"), FailureReason.Unauthorized);
                }
                Console.WriteLine($"Error fetching models! Status Code: {response.StatusCode}!");
                return Result<List<LCModelInferenceDto>>.Failure(Loc.T("LCErrors_ErrorFetchingData"));
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return Result<List<LCModelInferenceDto>>.Failure(Loc.T("LCErrors_ErrorFetchingData"));
            }
        }

        public async Task<Result<LCGridPageDataDto>> GetModelsForOverview(LCGridOptionsDto options)
        {
            try
            {
                var response = await _httpClient.PostAsJsonAsync("LungCancer/Models/Overview", options);
                if (response.IsSuccessStatusCode)
                {
                    var models = await response.Content.ReadFromJsonAsync<LCGridPageDataDto>() ?? new();
                    return Result<LCGridPageDataDto>.Success(models);
                }
                if (response.StatusCode == System.Net.HttpStatusCode.Unauthorized)
                {
                    return Result<LCGridPageDataDto>.Failure(Loc.T("LCErrors_UnauthorizedAccess"), FailureReason.Unauthorized);
                }

                Console.WriteLine($"Error fetching models! Status Code: {response.StatusCode}!");
                return Result<LCGridPageDataDto>.Failure(Loc.T("LCErrors_ErrorFetchingData"));
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return Result<LCGridPageDataDto>.Failure(Loc.T("LCErrors_ErrorFetchingData"));
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
                    return Result<LCInfoDto>.Failure(Loc.T("LCErrors_ErrorFetchingData"));
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result<LCInfoDto>.Failure(Loc.T("LCErrors_ErrorFetchingData"));
            }
        }

        public async Task<Result<LCInferenceResultDto>> PredictAsync(int id, IBrowserFile file)
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
                    var prediction = await response.Content.ReadFromJsonAsync<LCInferenceResultDto>();
                    return Result<LCInferenceResultDto>.Success(prediction!);
                }
                else
                {
                    var errorDetails = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"API FAILURE: {errorDetails}");
                    return Result<LCInferenceResultDto>.Failure(Loc.T("LCErrors_ErrorGeneric"));
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result<LCInferenceResultDto>.Failure(Loc.T("LCErrors_ErrorGeneric"));
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
                    return Result<int>.Failure(Loc.T("LCErrors_ErrorGeneric"));
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result<int>.Failure(Loc.T("LCErrors_ErrorGeneric"));
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
                    return Result<bool>.Failure(Loc.T("LCErrors_ErrorGeneric"));
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result<bool>.Failure(Loc.T("LCErrors_ErrorGeneric"));
            }
        }

        public async Task<Result> UpdateModelNameAsync(LCModelOverviewDto model)
        {
            try
            {
                var url = $"LungCancer/UpdateModelName?id={model.Id}";
                var request = new HttpRequestMessage(HttpMethod.Post, url);
                request.Content = JsonContent.Create(model.Name);
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
                    return Result.Failure(Loc.T("LCErrors_ErrorGeneric"));

                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result.Failure(Loc.T("LCErrors_ErrorGeneric"));
            }
        }
    }
}


