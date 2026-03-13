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

        public async Task<Result<LungCancerModel>> TrainModelAsync(LungCancerTrainingParams trainDto)
        {
            try
            {
                string url = "Python/LungCancer/Train";
                var request = new HttpRequestMessage(HttpMethod.Post, url);
                request.Content = JsonContent.Create(trainDto, options: _jsonOptions);
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
    }
}
