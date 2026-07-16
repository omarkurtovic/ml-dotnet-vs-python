var builder = DistributedApplication.CreateBuilder(args);

var storageRoot = Path.GetFullPath(Path.Combine(builder.Environment.ContentRootPath, "..", "storage"));

var pythonApi = builder.AddUvicornApp(
    name: "pythonapi",
    appDirectory: @"..\python-model-trainer",
    app: "main:app")
    .WithEnvironment("ML_STORAGE_ROOT", storageRoot);

var apiService = builder.AddProject<Projects.CSharpModelTrainerApi>("apiservice")
    .WithHttpHealthCheck("/health")
    .WithEnvironment("ML_STORAGE_ROOT", storageRoot)
    .WithReference(pythonApi);

builder.AddProject<Projects.WebApp>("webfrontend")
.WithExternalHttpEndpoints()
.WithHttpHealthCheck("/health")
.WithReference(apiService)
.WithReference(pythonApi)
.WaitFor(apiService);

builder.Build().Run();
