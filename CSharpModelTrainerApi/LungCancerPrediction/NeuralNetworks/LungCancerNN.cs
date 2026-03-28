using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.TensorExtensionMethods;
using static TorchSharp.torch.distributions;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using static TorchSharp.torch.utils;
using static TorchSharp.torch.utils.data;

namespace CSharpModelTrainerApi.LungCancerPrediction.NeuralNetworks
{
    public class LungCancerNN : nn.Module<Tensor, Tensor>
    {
        public LungCancerNN() : base("LungCancerCNN")
        {
            model = Sequential(
                ("conv2d", Conv2d(in_channels: 1, out_channels: 64, kernel_size: 3)),
                ("relu1", ReLU()),
                ("maxpooling2d1", MaxPool2d((2, 2))),
                ("conv2d2", Conv2d(in_channels: 64, out_channels: 64, kernel_size: 3)),
                ("relu2", ReLU()),
                ("maxpooling2d2", MaxPool2d((2, 2))),
                ("flatten", Flatten()),
                ("dense1", Linear(inputSize: 246016, outputSize: 16)),
                ("dense2", Linear(inputSize: 16, outputSize: 3))
            );

            RegisterComponents();
        }

        Sequential model;

        public override Tensor forward(Tensor input)
        {
            return model.call(input);
        }
    }
}
