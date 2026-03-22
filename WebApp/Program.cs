using Microsoft.AspNetCore.HttpOverrides;
using MudBlazor.Services;
using WebApp.Components;
using WebApp.LungCancerPrediction.ApiClients;
using WebApp.SentimentAnalysis.ApiClients;

var builder = WebApplication.CreateBuilder(args);

builder.AddServiceDefaults();

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();
builder.Services.AddMudServices();


#pragma warning disable EXTEXP0001
builder.Services.AddHttpClient<CSharpSentimentAnalysisApiClient>(client =>
{
    // This URL uses "https+http://" to indicate HTTPS is preferred over HTTP.
    // Learn more about service discovery scheme resolution at https://aka.ms/dotnet/sdschemes.
    client.BaseAddress = new("https+http://apiservice");
    client.Timeout = TimeSpan.FromMinutes(30);
}).RemoveAllResilienceHandlers();


builder.Services.AddHttpClient<PythonSentimentAnalysisApiClient>(client =>
{
    client.BaseAddress = new("https+http://pythonapi");
    client.Timeout = TimeSpan.FromMinutes(30);
}).RemoveAllResilienceHandlers();

builder.Services.AddHttpClient<CSharpLungCancerApiClient>(client =>
{
    // This URL uses "https+http://" to indicate HTTPS is preferred over HTTP.
    // Learn more about service discovery scheme resolution at https://aka.ms/dotnet/sdschemes.
    client.BaseAddress = new("https+http://apiservice");
    client.Timeout = TimeSpan.FromMinutes(30);
}).RemoveAllResilienceHandlers();


builder.Services.AddHttpClient<PythonLungCancerApiClient>(client =>
{
    client.BaseAddress = new("https+http://pythonapi");
    client.Timeout = TimeSpan.FromMinutes(30);
}).RemoveAllResilienceHandlers();
#pragma warning restore EXTEXP0001

var app = builder.Build();

app.MapDefaultEndpoints();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}


app.UseForwardedHeaders(new ForwardedHeadersOptions
{
    ForwardedHeaders = ForwardedHeaders.XForwardedFor | ForwardedHeaders.XForwardedProto
});

app.UseStatusCodePagesWithReExecute("/not-found", createScopeForStatusCodePages: true);
app.UseHttpsRedirection();

app.UseAntiforgery();

app.MapStaticAssets();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
