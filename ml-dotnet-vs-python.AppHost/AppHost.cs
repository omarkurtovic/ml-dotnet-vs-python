var builder = DistributedApplication.CreateBuilder(args);

var apiService = builder.AddProject<Projects.CSharpModelTrainerApi>("apiservice")
    .WithHttpHealthCheck("/health");

builder.AddProject<Projects.WebApp>("webfrontend")
    .WithExternalHttpEndpoints()
    .WithHttpHealthCheck("/health")
    .WithReference(apiService)
    .WaitFor(apiService);

builder.Build().Run();
