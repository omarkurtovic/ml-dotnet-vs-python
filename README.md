# ML.NET vs Python: A Comparative Analysis

## 🎯 Project Goals

Compare ML.NET (C#) and scikit-learn (Python) across:
- Model accuracy
- Training performance
- Prediction latency
- Deployment complexity
- Development experience

## 📋 Todo List

### Phase 1: Data & Python Baseline (Week 1)
- [ ] Find and download car price dataset (Kaggle)
- [ ] Exploratory data analysis (EDA) in Jupyter
- [ ] Data cleaning and preprocessing
- [ ] Train baseline Python model (Linear Regression)
- [ ] Train comparison model (Random Forest)
- [ ] Evaluate and save metrics (RMSE, R², training time)
- [ ] Save trained model (pickle/joblib)

### Phase 2: ML.NET Implementation (Week 2)
- [ ] Create .NET Console/API project
- [ ] Install ML.NET NuGet packages
- [ ] Load and preprocess data in C#
- [ ] Train ML.NET model (Linear Regression)
- [ ] Train ML.NET model (FastTree/Decision Tree)
- [ ] Evaluate and save metrics
- [ ] Save trained model (.zip)

### Phase 3: API Development (Week 3)
- [ ] Create ASP.NET Core Web API project
- [ ] Endpoint: `/predict/python` (calls Python model)
- [ ] Endpoint: `/predict/mlnet` (calls ML.NET model)
- [ ] Endpoint: `/predict/compare` (calls both, returns comparison)
- [ ] Add model loading on startup
- [ ] Add request validation
- [ ] Document API with Swagger

### Phase 4: Frontend (Week 4)
- [ ] Create Blazor Server app (or simple HTML page)
- [ ] Form: input car features (year, mileage, brand, etc.)
- [ ] Display predictions from both models side-by-side
- [ ] Show prediction time for each
- [ ] Add simple styling (MudBlazor or Bootstrap)

### Phase 5: Testing & Metrics (Week 5)
- [ ] Performance testing (100+ predictions, measure latency)
- [ ] Compare model sizes (.pkl vs .zip)
- [ ] Compare deployment complexity (Docker?)
- [ ] Memory usage comparison
- [ ] Create comparison tables and charts
- [ ] Document findings

### Phase 6: Documentation (Week 6)
- [ ] Write thesis introduction
- [ ] Document methodology
- [ ] Add results section with tables/graphs
- [ ] Conclusion and recommendations
- [ ] Code documentation (inline comments)
- [ ] Final README with setup instructions
- [ ] Create architecture diagram

## 🛠️ Tech Stack

**Python:**
- scikit-learn
- pandas, numpy
- Flask (for API wrapper if needed)

**C#/.NET:**
- ML.NET
- ASP.NET Core Web API
- Blazor (optional frontend)

**Deployment:**
- Docker (optional)
- GitHub Actions (optional CI/CD)

## 📊 Expected Deliverables

1. Trained models (Python + ML.NET)
2. Comparison API
3. Simple frontend demo
4. Performance metrics report
5. Written thesis (20-30 pages)
6. GitHub repository with full code

## 🚀 Quick Start

*(Add setup instructions once project is built)*

## 📝 License

MIT
