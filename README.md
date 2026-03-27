## C# vs. Python: Machine Learning Architecture

A web application built to benchmark and compare machine learning model training and inference between the .NET (C#) and Python ecosystems.
Developed as part of a Bachelor's Thesis focusing on Computer Vision.

Supported Machine Learning Domains:

- Computer Vision : Lung Cancer image classification using Convolutional Neural Networks (CNNs).

- Natural Language Processing : Sentiment Analysis on text data using various classification algorithms.



#### Features

- Model Training: Trigger model training on either the C# or Python backend from a single UI.

- Side-by-Side Inference: Test predictions using both models to compare outputs and latency.

- Model Management: View detailed performance metrics (accuracy, training time) and delete old models.

- Persistent Storage: All model metadata is tracked and stored in a unified SQLite database.



#### Live Demo

Check out the live demo here: [ml-dotnet-vs-python](http://49.13.166.156/lungCancer)



#### Tech Stack

- Blazor Server
- ASP.NET Core Web API
- SQLite + EF Core
- ML.NET (C#) and TorchSharp (C# for CNNs)
- FastAPI (Python)
- Scikit-Learn and Keras/PyTorch (Python)
- .NET Aspire for orchestration
- MudBlazor for UI components
- Hetzner Cloud (Ubuntu) for deployment