## C# vs. Python: Machine Learning Architecture

A web application built to benchmark and compare machine learning model training and inference between the .NET (C#) and Python ecosystems.
Developed as part of a Bachelor's Thesis focusing on Computer Vision. The program is focused on a computer vision task of Lung Cancer detection.

#### Features

- Model Training: Trigger model training on either the C# or Python backend from a single UI.

- Side-by-Side Inference: Test predictions using both models to compare outputs and latency.

- Model Management: View detailed performance metrics (accuracy, training time) and delete old models.

- Persistent Storage: All model metadata is tracked and stored in a unified SQLite database.

#### Live Demo

Check out the live demo here: [ml-dotnet-vs-python](http://49.13.166.156)

#### Screenshots

<img width="2557" height="1284" alt="image" src="https://github.com/user-attachments/assets/548e3dd5-bbe2-4d61-8d8b-c4362f98dda4" />
<img width="2557" height="1281" alt="image" src="https://github.com/user-attachments/assets/7f34b95d-7ee2-4897-9fd5-1d2302f24511" />
<img width="2556" height="1284" alt="image" src="https://github.com/user-attachments/assets/c7699ec7-f505-43d6-8351-1101dc114298" />


#### Tech Stack

- Blazor Server
- ASP.NET Core Web API
- SQLite + EF Core
- TorchSharp)
- FastAPI (Python)
- PyTorch
- .NET Aspire for orchestration
- MudBlazor for UI components
- Hetzner Cloud (Ubuntu) for deployment
