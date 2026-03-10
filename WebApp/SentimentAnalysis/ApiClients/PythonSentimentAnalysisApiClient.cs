using SharedCL.SentimentAnalysis.Models;
using SharedCL.Shared.Models;
using System.Net.Http.Headers;
using System.Text.Json;

namespace WebApp.SentimentAnalysis.ApiClients
{
    public class PythonSentimentAnalysisApiClient(HttpClient httpClient)
    {
        private readonly HttpClient _httpClient = httpClient;

        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        public async Task<Result<SentimentAnalysisModel>> TrainModelAsync(SentimentAnalysisTrainingParams trainDto)
        {
            try
            {
                string url = "Python/SentimentAnalysis/Train";
                var request = new HttpRequestMessage(HttpMethod.Post, url);
                request.Content = JsonContent.Create(trainDto, options: _jsonOptions);
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
    }
}
