using TorchSharp;

namespace CSharpModelTrainerApi.LungCancerPrediction.Transforms
{
    public class AffineTransform : torchvision.ITransform
    {
        public torch.Tensor call(torch.Tensor input1)
        {
            int angle = torch.randint(low: -25, high: 26, size: [1], dtype: torch.int32).item<int>();
            int xTranslation = torch.randint(low: -15, high: 16, size: [1], dtype: torch.int32).item<int>();
            int yTranslation = torch.randint(low: -15, high: 16, size: [1], dtype: torch.int32).item<int>();
            int xShear = torch.randint(low: -15, high: 16, size: [1], dtype: torch.int32).item<int>();
            int yShear = torch.randint(low: -15, high: 16, size: [1], dtype: torch.int32).item<int>();
            float scale = torch.rand(size: [1], dtype: torch.float32).item<float>() * 0.2f + 0.9f;
            return torchvision.transforms.functional.affine(input1, shear: [xShear, yShear], angle: angle, translate: [xTranslation, yTranslation], scale: scale);
        }
    }
}
