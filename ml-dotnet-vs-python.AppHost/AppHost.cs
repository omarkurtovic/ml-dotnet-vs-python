var builder = DistributedApplication.CreateBuilder(args);


var apiService = builder.AddProject<Projects.CSharpModelTrainerApi>("apiservice")
    .WithHttpHealthCheck("/health");

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
.WaitFor(apiService);

builder.Build().Run();
