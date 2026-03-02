namespace WebApp.SharedKernel
{
    public class PredictionModel
    {
        public string Name { get; set; } = null!;
        public PredictionModelType Model { get; set; }
    }
    public enum PredictionModelType
    {
        CSharp,
        Python
    }
}
