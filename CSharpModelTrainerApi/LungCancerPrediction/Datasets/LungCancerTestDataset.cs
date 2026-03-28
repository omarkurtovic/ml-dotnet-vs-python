using CSharpModelTrainerApi.LungCancerPrediction.Services;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;

namespace CSharpModelTrainerApi.LungCancerPrediction.Datasets
{
    public class LungCancerTestDataset : Dataset
    {
        private readonly List<Tensor> _images = [];
        private readonly List<long> _labels = [];

        public LungCancerTestDataset(string dataDirectory)
        {
            var categories = new List<string> { "Bengin cases", "Malignant cases", "Normal cases" };

            for (int i = 0; i < categories.Count; ++i)
            {
                var path = Path.Join(dataDirectory, categories[i]);
                var files = Directory.GetFiles(path);
                for (int j = (int)(files.Length * 0.75); j < files.Length; ++j)
                {
                    _images.Add(ImageLoader.ImagePathToTensor(files[j]));
                    _labels.Add(i);
                }
            }
        }
        public override long Count => _images.Count;

        public override Dictionary<string, Tensor> GetTensor(long index)
        {
            var label = _labels[(int)index];
            var image = _images[(int)index];

            return new Dictionary<string, Tensor>
            {
                ["image"] = image,
                ["label"] = torch.tensor(label)
            };
        }
    }
}
