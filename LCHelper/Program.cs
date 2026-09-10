
using CSharpModelTrainerApi.LungCancerPrediction.Datasets;
using CSharpModelTrainerApi.LungCancerPrediction.NeuralNetworks;
using SkiaSharp;
using System.Runtime.InteropServices;
using TorchSharp;


while (true)
{
    Console.WriteLine("Please select an option:");
    Console.WriteLine("1. Transform images");
    Console.WriteLine("2. Show model summary");
    Console.WriteLine("3. Exit");
    var choice = Console.ReadLine();
    if(choice == "1")
    {
        Console.WriteLine("Transforming images...");
        TransformImages();
        break;
    }
    else if (choice == "2")
    {
        ShowModelSummary();
        Console.WriteLine("Showing model summary...");
    }
    else if (choice == "3")
    {
        break;
    }
}

void ShowModelSummary()
{
    var model = new LungCancerNN();
    long total = 0;
    foreach (var (name, param) in model.named_parameters())
    {
        long count = param.numel();
        total += count;
        Console.WriteLine($"{name}\t{string.Join("×", param.shape)}\t{count}");
    }
    Console.WriteLine($"Ukupno: {total}");
}

void TransformImages()
{
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

    var trainingData = new LungCancerTrainDataset(dataDirectory, withAugmentation: true);

    for (int i = 0; i < 10; ++i)
    {
        var sample = trainingData.GetTensor(0)["image"];
        var tensorNonNormalized = sample.clamp(0, 1).mul(255F).to(torch.ScalarType.Byte);
        var bytes = tensorNonNormalized.data<byte>().ToArray();

        using SKBitmap bitmap = new(256, 256, SKColorType.Gray8, SKAlphaType.Opaque);

        IntPtr pixelBuffer = Marshal.AllocHGlobal(bytes.Length);
        Marshal.Copy(bytes, 0, pixelBuffer, bytes.Length);

        bitmap.InstallPixels(new SKImageInfo(256, 256, SKColorType.Gray8, SKAlphaType.Opaque), pixelBuffer);
        var data = SKImage.FromBitmap(bitmap).Encode(SKEncodedImageFormat.Png, 100);
        using var stream = File.OpenWrite(Path.Combine(transformationDirectory, $"transformed_image_{i}.png"));
        data.SaveTo(stream);

        Marshal.FreeHGlobal(pixelBuffer);
    }
}