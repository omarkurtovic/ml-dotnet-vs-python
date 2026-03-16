Need to fix model saving methodology

# ML .NET vs Python

A web app powered by Azure for training and comparing machine learning models, built with .NET 10.

## Overview

The app lets you train, view, and delete machine learning models for two use cases:
- **Sentiment Analysis**
- **Lung Cancer Detection**

## Projects

| Project | Description |
|---|---|
| `WebApp` | Blazor frontend — view models, train new ones, delete them |
| `CSharpModelTrainerApi` | C# ASP.NET Core API — ML model training and inference endpoints |
| `python-model-trainer` | Python FastAPI — alternative ML model training and inference endpoints |
| `SharedCL` | Shared class library used across C# projects |
| `ml-dotnet-vs-python.AppHost` | .NET Aspire app host |
| `ml-dotnet-vs-python.ServiceDefaults` | .NET Aspire service defaults |

## Tech Stack

- **Frontend:** Blazor (.NET 10)
- **C# API:** ASP.NET Core / ML.NET
- **Python API:** FastAPI / scikit-learn
- **Infrastructure:** Azure (via .NET Aspire)