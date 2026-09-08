using SharedCL;
using System.Net.Http.Json;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class PythonLCApiClient(HttpClient httpClient)
    {
        private readonly HttpClient _httpClient = httpClient;

        public async Task StartTrainingAsync(int modelId, LCTrainingParamsDto trainingParams)
        {
            var response = await _httpClient.PostAsJsonAsync(
                $"Python/LungCancer/Train?model_id={modelId}", trainingParams);
            response.EnsureSuccessStatusCode();
        }

        public async Task<List<LCTrainingProgressDto>?> GetTrainingProgressAsync(int modelId)
        {
            return await _httpClient.GetFromJsonAsync<List<LCTrainingProgressDto>>(
                $"Python/LungCancer/Train/{modelId}");
        }
    }
}
