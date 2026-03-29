using CSharpModelTrainerApi.LungCancerPrediction.Services;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;

namespace CSharpModelTrainerApi.LungCancerPrediction.Datasets
{
    public class LungCancerTrainDataset : Dataset
    {
        private readonly List<Tensor> _images = [];
        private readonly List<long> _labels = [];

        public LungCancerTrainDataset(string dataDirectory)
        {
            var categories = new List<string> { "Bengin cases", "Malignant cases", "Normal cases" };

            for (int i = 0; i < categories.Count; ++i)
            {
                var path = Path.Join(dataDirectory, categories[i]);
                var files = Directory.GetFiles(path);
                int categoryImageCount = (int)(files.Length * 0.75);

                for (int j = 0; j < categoryImageCount; ++j)
                {
                    var tensor = ImageLoader.ImagePathToTensor(files[j]);
                    _images.Add(tensor);
                    _labels.Add(i);

                    _images.Add(tensor.flip([2]).clone());
                    _labels.Add(i);

                    _images.Add(tensor.flip([1]).clone());
                    _labels.Add(i);
                }
            }
        }
        public override long Count => _images.Count;

        public override Dictionary<string, Tensor> GetTensor(long index)
        {
            var image = _images[(int)index];
            var label = _labels[(int)index];

            return new Dictionary<string, Tensor>
            {
                ["image"] = image,
                ["label"] = torch.tensor(label)
            };
        }
        public float[] GetClassWeights()
        {
            int total = _labels.Count;
            int numClasses = 3;
            var counts = _labels.GroupBy(l => l).ToDictionary(g => g.Key, g => g.Count());
            return Enumerable.Range(0, numClasses)
                .Select(i => (float)total / (numClasses * counts[(long)i]))
                .ToArray();
        }
    }
}
