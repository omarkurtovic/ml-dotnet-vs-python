using CSharpModelTrainerApi.Database;
using CSharpModelTrainerApi.LungCancerPrediction.Services;
using CSharpModelTrainerApi.LungCancerPrediction.Workers;
using CSharpModelTrainerApi.Services;
using Microsoft.EntityFrameworkCore;
using System;

var builder = WebApplication.CreateBuilder(args);

// Add service defaults & Aspire client integrations.
builder.AddServiceDefaults();

// Add services to the container.
builder.Services.AddProblemDetails();

builder.Services.AddControllers();
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();

ConfigureDatabase(builder.Services, builder.Configuration);

builder.Services.AddScoped<LCRepository>();
builder.Services.AddSingleton<LCPredictionService>();
#pragma warning disable EXTEXP0001
builder.Services.AddHttpClient<PythonLCApiClient>(client =>
{
    client.BaseAddress = new("https+http://pythonapi");
    client.Timeout = TimeSpan.FromSeconds(30);
}).RemoveAllResilienceHandlers();
#pragma warning restore EXTEXP0001

builder.Services.AddSingleton<PathResolver>();
builder.Services.AddSingleton<HardwareInfoService>();

builder.Services.AddSingleton<TrainingQueue>();
builder.Services.AddHostedService<TrainingWorker>();

var app = builder.Build();

app.Services.GetRequiredService<PathResolver>().InitializeStorage();

app.MapDefaultEndpoints();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.MapControllers();

InitializeDatabase(app);

app.Run();





void ConfigureDatabase(IServiceCollection services, IConfiguration configuration)
{
    var storageRoot = Environment.GetEnvironmentVariable("ML_STORAGE_ROOT")
        ?? configuration["Storage:Root"];

    if (string.IsNullOrWhiteSpace(storageRoot))
        throw new InvalidOperationException("ML_STORAGE_ROOT is not configured.");

    storageRoot = Path.GetFullPath(storageRoot);
    Directory.CreateDirectory(storageRoot);

    var dbPath = Path.Combine(storageRoot, "app.db");
    services.AddDbContext<AppDbContext>(options =>
        options.UseSqlite($"Data Source={dbPath}"));
}

void InitializeDatabase(WebApplication app)
{
    using var scope = app.Services.CreateScope();
    var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
    db.Database.Migrate();
}
