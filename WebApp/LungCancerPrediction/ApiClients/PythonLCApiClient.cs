using Microsoft.AspNetCore.Components.Forms;
using SharedCL;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;

namespace WebApp.LungCancerPrediction.ApiClients
{
    public class PythonLCApiClient(HttpClient httpClient)
    {
        private readonly HttpClient _httpClient = httpClient;

        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        public async Task<Result<LCDto>> TrainModelAsync(LCTrainingParamsDto trainDto)
        {
            try
            {
                string url = "Python/LungCancer/Train";
                var request = new HttpRequestMessage(HttpMethod.Post, url)
                {
                    Content = JsonContent.Create(trainDto, options: _jsonOptions)
                };
                request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
                var response = await _httpClient.SendAsync(request);
                if (response.IsSuccessStatusCode)
                {
                    var modelDto = await response.Content.ReadFromJsonAsync<LCDto>();
                    return Result<LCDto>.Success(modelDto!);
                }
                else
                {
                    var errorDetails = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"API FAILURE: {errorDetails}");
                    return Result<LCDto>.Failure(Loc.T("LCErrors_ErrorGeneric"));
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result<LCDto>.Failure(Loc.T("LCErrors_ErrorGeneric"));
            }
        }

        public async Task<Result<LCPredictionDto>> PredictAsync(string modelName, IBrowserFile file)
        {
            try
            {
                string url = $"Python/LungCancer/Predict?model_name={modelName}";
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
                    return Result<LCPredictionDto>.Failure(Loc.T("LCErrors_ErrorGeneric"));
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API FAILURE: {ex.Message}");
                return Result<LCPredictionDto>.Failure(Loc.T("LCErrors_ErrorGeneric"));
            }
        }
    }
}
