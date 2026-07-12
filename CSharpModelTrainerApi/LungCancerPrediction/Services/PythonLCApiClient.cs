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

        public async Task<LCInfoDto?> GetTrainingInfoAsync(int modelId)
        {
            return await _httpClient.GetFromJsonAsync<LCInfoDto>(
                $"Python/LungCancer/Train/{modelId}");
        }
    }
}
