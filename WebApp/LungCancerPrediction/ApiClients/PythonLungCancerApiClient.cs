using SharedCL.LungCancerPrediction.Dtos;
using SharedCL.LungCancerPrediction.Models;
using SharedCL.SentimentAnalysis.Models;
using SharedCL.Shared.Models;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;

namespace WebApp.LungCancerPrediction.ApiClients
{
    public class PythonLungCancerApiClient(HttpClient httpClient)
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
    }
}
