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

        public async Task<Result<int>> TrainModelAsync(LCTrainingParamsDto trainDto)
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
    }
}
