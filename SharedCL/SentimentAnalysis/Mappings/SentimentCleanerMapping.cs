using Microsoft.ML.Transforms;
using SharedCL.CarValuePrediction.Models;
using SharedCL.SentimentAnalysis.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL.SentimentAnalysis.Mappings
{
    [CustomMappingFactoryAttribute("SentimentCleaner")]
    public class SentimentCleanerMapping : CustomMappingFactory<SentimentData, SentimentClean>
    {
        public override Action<SentimentData, SentimentClean> GetMapping()
        {
            return (input, output) =>
            {
                output.SentimentValue = input.Sentiment.Equals("positive", StringComparison.OrdinalIgnoreCase);
            };
        }
    }
}
