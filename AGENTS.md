# AGENTS.md

## Architecture

Three-service .NET Aspire app comparing C# vs Python ML for lung cancer detection:

- **WebApp** (`WebApp/`) - Blazor Server UI (port 5001), references SharedCL
- **CSharpModelTrainerApi** (`CSharpModelTrainerApi/`) - ASP.NET Core Web API (port 5000), TorchSharp/ML.NET inference & training, SQLite via EF Core
- **python-model-trainer** (`python-model-trainer/`) - FastAPI + PyTorch (port 8000)
- **SharedCL** (`SharedCL/`) - shared C# library (ML.NET data types)
- **AppHost** (`ml-dotnet-vs-python.AppHost/`) - Aspire orchestrator, wires all three services together

All .NET projects target **net10.0**.

## Commands

```bash
# Run everything locally via Aspire (starts all 3 services)
dotnet run --project ml-dotnet-vs-python.AppHost

# Build
dotnet build

# Run C# tests (NUnit)
dotnet test CSharpTests/CSharpTests.csproj

# Run Python tests
cd python-model-trainer && python -m pytest Tests/
```

## Python Setup (one-time)

```bash
cd python-model-trainer
python3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

PyTorch installs from CUDA index (`cu128`). On Linux, also `apt-get install -y libgl1` (required by opencv-python-headless).

## Key Conventions

- `storage/` contains large data and model files; it is deployed via `scp`, not committed
- DB file: `app.db` in CSharpModelTrainerApi output (`bin/Debug/net10.0/`)
- `ML_STORAGE_ROOT` points both services to the shared data/model storage; Aspire sets it locally and systemd sets it in production
- No linter/formatter/typecheck config exists; no CI workflows yet
