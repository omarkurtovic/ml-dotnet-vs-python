
using CSharpModelTrainerApi.LungCancerPrediction.Datasets;


var storageRoot = Path.GetFullPath("storage");
var dataDirectory =  Path.Combine(storageRoot, "data", "lung-cancer-prediction");
var trainingData = new LungCancerTrainDataset(dataDirectory, withTransforms: true);