using CSharpModelTrainerApi.Database;
using CSharpModelTrainerApi.LungCancerPrediction.Services;
using CSharpModelTrainerApi.Shared;
using CSharpModelTrainerApi.SentimentAnalysis.Services;
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

ConfigureDatabase(builder.Services, builder.Environment);

builder.AddAzureBlobServiceClient(connectionName: "blobs"); 
builder.Services.AddSingleton<BlobService>();

builder.Services.AddSingleton<SentimentAnalysisModelTrainer>();
builder.Services.AddScoped<SentimentAnalysisRepository>();
builder.Services.AddSingleton<SentimentAnalysisPredictionServices>();

builder.Services.AddSingleton<LungCancerModelTrainer>();
builder.Services.AddScoped<LungCancerModelRepository>();
builder.Services.AddSingleton<LungCancerPredictionService>();

var app = builder.Build();

app.MapDefaultEndpoints();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.MapControllers();

InitializeDatabase(app);

app.Run();





void ConfigureDatabase(IServiceCollection services, IWebHostEnvironment env)
{
    var dbPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "app.db");
    services.AddDbContext<AppDbContext>(options =>
        options.UseSqlite($"Data Source={dbPath}"));
}

void InitializeDatabase(WebApplication app)
{

    using var scope = app.Services.CreateScope();
    var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
    db.Database.Migrate();
}