using Microsoft.ML.Transforms;

namespace SharedCL
{
    [CustomMappingFactoryAttribute("SentimentCleaner")]
    public class SentimentCleanerMapping : CustomMappingFactory<SADataDto, SACleanDto>
    {
        public override Action<SADataDto, SACleanDto> GetMapping()
        {
            return (input, output) =>
            {
                output.SentimentValue = input.Sentiment.Equals("positive", StringComparison.OrdinalIgnoreCase);
            };
        }
    }
}
