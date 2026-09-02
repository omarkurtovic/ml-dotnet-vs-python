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
        private readonly torchvision.ITransform? transformPipeline = null;
        private readonly string[] Categories = ["Bengin cases", "Malignant cases", "Normal cases"];

        public LungCancerTrainDataset(string dataDirectory, int? maxImagesPerCategory = null, bool withAugmentation = false)
        {
            this.transformPipeline = withAugmentation ? torchvision.transforms.Compose(
            [
                new Transforms.AffineTransform(),
                torchvision.transforms.ColorJitter(brightness: 0.2f, contrast: 0.2f, saturation: 0, hue: 0),
            ]) : null;

            for (int i = 0; i < Categories.Length; ++i)
            {
                var path = Path.Join(dataDirectory, Categories[i]);
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
                }
            }
        }

        public LungCancerTrainDataset(List<Tensor> images, List<long> labels, string[] categories)
        {
            Images = images;
            Labels = labels;
            Categories = categories;
        }

        public override long Count => Images.Count;


        public override Dictionary<string, Tensor> GetTensor(long index)
        {
            var image = Images[(int)index];
            var label = Labels[(int)index];

            return new Dictionary<string, Tensor>
            {
                ["image"] = transformPipeline?.call(image) ?? image,
                ["label"] = torch.tensor(label)
            };
        }

        public float[] GetClassWeights()
        {
            if (Categories.Length == 0)
            {
                return [];
            }

            if (Labels.Count == 0)
            {
                return new float[Categories.Length];
            }

            int totalLabels = Labels.Count;
            float[] classWeights = new float[Categories.Length];

            for (int i = 0; i < classWeights.Length; ++i)
            {
                int classCount = Labels.Count(l => l == i);
                if (classCount == 0)
                {
                    classWeights[i] = 0f;
                }
                else
                {
                    classWeights[i] = totalLabels / ((float)Categories.Length * classCount);
                }
            }

            return classWeights;
        }
    }
}
