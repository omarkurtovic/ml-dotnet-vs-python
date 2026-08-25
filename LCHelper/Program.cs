
using CSharpModelTrainerApi.LungCancerPrediction.Datasets;
using TorchSharp;


var dataDirectory = @"C:\Users\Administrator\source\repos\omarkurtovic\ml-dotnet-vs-python\storage\data\lung-cancer-prediction";
var transformationDirectory = @"C:\Users\Administrator\source\repos\omarkurtovic\ml-dotnet-vs-python\storage\data\lung-cancer-prediction\transformed_images";

if (!Directory.Exists(transformationDirectory))
{
    Directory.CreateDirectory(transformationDirectory);
}

DirectoryInfo di = new(transformationDirectory);
foreach (FileInfo file in di.GetFiles())
{
    file.Delete();
}
foreach (DirectoryInfo dir in di.GetDirectories())
{
    dir.Delete(true);
}

var trainingData = new LungCancerTrainDataset(dataDirectory, withTransforms: true);

for(int i = 0; i < 10; ++i)
{
    var tensor = trainingData.GetTensor(i);
    foreach(KeyValuePair<string, torch.Tensor> kvp in tensor)
    {
        var tensorNonNormalized = kvp.Value.clamp(0, 1).mul(255F).to(torch.ScalarType.Byte);

        torchvision.io.write_png(tensorNonNormalized, Path.Combine(transformationDirectory, $"image_{i}.png"));
    }
}