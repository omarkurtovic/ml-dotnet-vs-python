# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Bachelor's thesis benchmark comparing ML training/inference between the .NET and Python ecosystems on one computer-vision task: lung cancer classification (IQ-OTH/NCCD CT scans, 3 classes). The C# and Python trainers are **deliberate mirrors of each other** — the comparison is only meaningful if they stay identical. See "Cross-language parity" below.

## Commands

```powershell
# Run everything (Aspire starts C# API, Python API, and Blazor UI together)
dotnet run --project ml-dotnet-vs-python.AppHost

dotnet build                                    # whole solution
dotnet test CSharpTests/CSharpTests.csproj      # NUnit
dotnet test CSharpTests/CSharpTests.csproj --filter "FullyQualifiedName~GetClassWeights_UnbalancedClasses"

# Python (from python-model-trainer/)
python -m pytest Tests/
python -m pytest Tests/test_datasets.py::test_get_class_weights_missing_class_avoids_division_by_zero
```

One-time Python setup (Python 3.12; `.venv` is what Aspire's `AddUvicornApp` launches):

```powershell
cd python-model-trainer
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

PyTorch resolves from the CUDA index (`cu128`). On Linux also `apt-get install -y libgl1` for opencv.

Do not run the services individually unless debugging one in isolation — the WebApp and the C# API resolve each other through Aspire service discovery (`https+http://apiservice`, `https+http://pythonapi`), which only works under the AppHost. Ports 5000/5001/8000 in `DEPLOYMENT.md` are the systemd/production layout, not the local one.

EF Core migrations are applied automatically at C# API startup (`Program.cs` → `db.Database.Migrate()`); no manual `dotnet ef database update` step.

There is no linter, formatter, typecheck config, or CI workflow in this repo.

## Architecture

| Project | Role |
|---|---|
| `ml-dotnet-vs-python.AppHost/` | Aspire orchestrator; wires all three services and injects `ML_STORAGE_ROOT` |
| `WebApp/` | Blazor Server UI (MudBlazor + ApexCharts); calls **both** APIs directly |
| `CSharpModelTrainerApi/` | ASP.NET Core API; TorchSharp training/inference, owns the SQLite DB |
| `python-model-trainer/` | FastAPI + PyTorch; training/inference only, no DB |
| `SharedCL/` | DTOs + `Result<T>` shared by WebApp and the C# API |
| `ml-dotnet-vs-python.ServiceDefaults/` | Standard Aspire health/telemetry/resilience wiring |

All .NET projects target **net10.0**. `csharp_model_trainer/` is a legacy standalone console trainer, not in the solution — ignore it unless explicitly asked.

### The DB is owned only by the C# API

There is one SQLite database (`app.db` under `ML_STORAGE_ROOT`), and only `CSharpModelTrainerApi` touches it. This produces an asymmetry that explains most of the odd-looking code:

- **C# training**: `LungCancerController.Train` saves the model row, then enqueues to `TrainingQueue`; `TrainingWorker` (a `BackgroundService`) runs `LCTrainer`, which writes each epoch to the DB as it completes.
- **Python training**: the controller POSTs to the Python API and returns. Python runs training on a thread and holds progress in an **in-memory dict** (`_training_state` in `lc_controller.py`) — it never persists anything but the `.dat` weights file.
- Python epoch data reaches the DB only as a side effect of `GET /LungCancer/Models/Info/{id}`: for Python models that endpoint polls the Python API, appends any newly-completed epoch via `LCRepository.AddEpochData`, and syncs status. The Blazor UI polling this endpoint *is* the persistence mechanism. Python progress is lost if the Python process restarts mid-training.

Repository methods return `Result<T>` (`SharedCL/Models/Result.cs`) rather than throwing; controllers map `!IsSuccess` to `BadRequest`.

### Storage layout

`ML_STORAGE_ROOT` is **required** by both API services (they throw at startup without it) and is the single source of truth for data and model paths — mirrored in `CSharpModelTrainerApi/Services/PathResolver.cs` and `LungCancerPrediction/services.py::PathResolver`. Aspire sets it to `<repo>/storage` locally; systemd sets `/opt/app/storage` in production.

```
storage/
  app.db
  data/lung-cancer-prediction/{Bengin cases,Malignant cases,Normal cases}/
  models/lung-cancer-prediction/{csharp,python}/<ModelName>.dat
```

`storage/` is gitignored and shipped by `scp` (see `DEPLOYMENT.md`). Model files are keyed by **model name**, not id — renaming a model (`UpdateModelName`) moves the file on disk, and deleting a model deletes it.

## Cross-language parity

`LCTrainer.cs` / `lc_controller.py`, `LungCancerNN.cs` / `neural_networks.py`, `Datasets/*.cs` / `datasets.py`, and `PathResolver`/`ImageLoader`/`HardwareInfoService` in each stack are line-for-line counterparts. **A change to either side must be mirrored on the other**, or the benchmark numbers stop being comparable. The pinned values on both sides:

- Seed 42 (`manual_seed` + CUDA seeds), Adam `lr=1e-4`, batch size 8, `clip_grad_norm_(max_norm=1.0)`, `CrossEntropyLoss` with computed class weights
- CNN: Conv(1→64,k3) → ReLU → MaxPool2 → Conv(64→64,k3) → ReLU → MaxPool2 → Flatten → Linear(246016→16) → Linear(16→3). The 246016 is hard-coded for 256×256 grayscale input — changing `ImageLoader.IMAGE_SIZE` requires recomputing it in both `LungCancerNN` classes
- Class index order is **Benign=0, Malignant=1, Normal=2**, matching the directory list `["Bengin cases", "Malignant cases", "Normal cases"]` — the "Bengin" misspelling is the actual dataset directory name and must not be "fixed". Note `ClassificationReport`/`classification_report` assign metric fields by that index order, while `LCTrainer.cs`'s `EpochData` declares Normal before Malignant — go by the index, not the declaration order
- Split is positional, not random: first 75% of each category's directory listing is train, the remainder is validation. Augmentation (`withFlips`) adds a horizontal and a vertical flip per training image
- Timing counts **training only** (validation is excluded from `TrainingTimeInSeconds`), accumulated across epochs

Device selection is `cuda` if available else `cpu` on both sides. TorchSharp's native backend is chosen by RID in `CSharpModelTrainerApi.csproj`: `TorchSharp-cuda-windows` for an empty or `win-x64` RID, `TorchSharp-cpu` only when publishing with `-r linux-x64`.

## Conventions

- **DTOs**: `SharedCL/LungCancerPrediction/Dtos/*` are the contract for both APIs. Python re-declares them in `LungCancerPrediction/models.py` with camelCase field names; the C# HTTP clients serialize camelCase to match. Domain enums (`CSharpModelTrainerApi/Enums/`) are separate types from the DTO enums (`SharedCL/Enums/`) and are cast between — keep the numeric values aligned.
- **Localization**: `WebApp/Loc.cs` is a hand-written static dictionary (`en-US`, `bs-BA`) accessed as `Loc.T("Key")` — not resx, not `IStringLocalizer`. Add every new key to both culture blocks. Supported cultures are also registered in `WebApp/Program.cs`.
- Some API validation/error strings are Bosnian; UI-facing text goes through `Loc.T` instead.
- Tests cover only `GetClassWeights` on both sides, and the C# and Python test suites are intentional mirrors of each other — add to both.
