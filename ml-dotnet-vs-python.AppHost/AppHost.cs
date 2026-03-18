var builder = DistributedApplication.CreateBuilder(args);

var blobs = builder.AddConnectionString("blobs");

var isProduction = builder.Environment.EnvironmentName == "Production";

var apiService = builder.AddProject<Projects.CSharpModelTrainerApi>("apiservice")
    .WithHttpHealthCheck("/health")
    .WithReference(blobs)
    .WithEnvironment("TRAINING_DISABLED", isProduction ? "true" : "false");

// py -3.12 -m venv .venv
// .venv\Scripts\pip.exe install -r .\requirements.txt

var pythonApi = builder.AddUvicornApp(
name: "pythonapi",
appDirectory: @"..\python-model-trainer",
app: "main:app");

builder.AddProject<Projects.WebApp>("webfrontend")
.WithExternalHttpEndpoints()
.WithHttpHealthCheck("/health")
.WithReference(apiService)
.WithReference(pythonApi)
.WithEnvironment("TRAINING_DISABLED", isProduction ? "true" : "false")
.WaitFor(apiService);

builder.Build().Run();
