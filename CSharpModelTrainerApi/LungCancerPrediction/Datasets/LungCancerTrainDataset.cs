using CSharpModelTrainerApi.LungCancerPrediction.Services;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;

namespace CSharpModelTrainerApi.LungCancerPrediction.Datasets
{
    public class LungCancerTrainDataset : Dataset
    {
        private readonly List<Tensor> Images = [];
        private readonly List<long> Labels = [];

        public LungCancerTrainDataset(bool withFlips, string dataDirectory, int? maxImagesPerCategory = null)
        {
            var categories = new List<string> { "Bengin cases", "Malignant cases", "Normal cases" };

            for (int i = 0; i < categories.Count; ++i)
            {
                var path = Path.Join(dataDirectory, categories[i]);
                var files = Directory.GetFiles(path);
                int categoryImageCount = (int)(files.Length * 0.75);

                if (maxImagesPerCategory.HasValue)
                {
                    categoryImageCount = Math.Min(categoryImageCount, maxImagesPerCategory.Value);
                }

                for (int j = 0; j < categoryImageCount; ++j)
                {
                    var tensor = ImageLoader.ImagePathToTensor(files[j]);
                    Images.Add(tensor);
                    Labels.Add(i);

                    if (withFlips)
                    {
                        Images.Add(tensor.flip([2]).clone());
                        Labels.Add(i);

                        Images.Add(tensor.flip([1]).clone());
                        Labels.Add(i);
                    }
                }
            }
        }
        public override long Count => Images.Count;

        public override Dictionary<string, Tensor> GetTensor(long index)
        {
            var image = Images[(int)index];
            var label = Labels[(int)index];

            return new Dictionary<string, Tensor>
            {
                ["image"] = image,
                ["label"] = torch.tensor(label)
            };
        }
        public float[] GetClassWeights()
        {
            int totalLabels = Labels.Count;
            float[] result = new float[3];
            for(int i = 0; i < result.Length; ++i)
            {
                int classCount = Labels.Count(l => l == i);
                result[i] = totalLabels / (3.0f * classCount);
            }

            return result;
        }
    }
}
