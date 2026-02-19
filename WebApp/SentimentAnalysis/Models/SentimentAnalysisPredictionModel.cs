namespace WebApp.SentimentAnalysis.Models
{
    public class SentimentAnalysisPredictionModel
    {
        public string Name { get; set; } = null!;
        public SentimentAnalysisPredictionModelType Model { get; set; }
    }
    public enum SentimentAnalysisPredictionModelType
    {
        CSharp,
        Python
    }
}
