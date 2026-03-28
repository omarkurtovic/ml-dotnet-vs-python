using SixLabors.ImageSharp.Processing;
using SkiaSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace CSharpModelTrainerApi.LungCancerPrediction.Services
{
    public class ImageLoader
    {
        public static Tensor ImagePathToTensor(string imagePath)
        {
            int imageSize = 256;
            using SKBitmap bitmap = SKBitmap.Decode(imagePath);
            using var resized = bitmap.Resize(new SKImageInfo(imageSize, imageSize, SKColorType.Gray8), SKFilterQuality.Medium);

            var pixelBytes = resized.Bytes; // single copy

            return torch.tensor(pixelBytes, dtype: ScalarType.Int8)
                        .to(ScalarType.Float32)
                        .div(255F)
                        .reshape(1, imageSize, imageSize);
        }

        public static async Task<Tensor> FormFileImageToTensor(IFormFile file)
        {
            const int imgSize = 256;

            using var stream = file.OpenReadStream();
            using var image = await SixLabors.ImageSharp.Image.LoadAsync<SixLabors.ImageSharp.PixelFormats.L8>(stream);
            image.Mutate(x => x.Resize(imgSize, imgSize));

            var floatData = new float[imgSize * imgSize];
            for (int y = 0; y < imgSize; y++)
                for (int x = 0; x < imgSize; x++)
                    floatData[y * imgSize + x] = image[x, y].PackedValue / 255.0f;

            return torch.tensor(floatData).reshape(1, 1, imgSize, imgSize);
        }
    }
}
