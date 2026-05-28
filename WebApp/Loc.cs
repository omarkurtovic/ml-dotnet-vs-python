using System.Globalization;

namespace WebApp;

public static class Loc
{
    private static readonly Dictionary<string, Dictionary<string, string>> Translations = new()
    {
        ["en-US"] = new()
        {
            ["LCHome_Title"] = "Lung Cancer Analysis",
            ["LCHome_Description"] = "This platform is used for training, analyzing, and directly comparing machine learning models developed in C# and Python programming languages ​​using IQOTHNCCD CT diagnostics.",
            ["LCHome_TrainModel"] = "Train Model",
            ["LCHome_TrainModelDescription" ] = "Configure and start the training process of a new model. Set the number of epochs, image augmentation (rotation), and select the desired backend system (C# or Python).",
            ["LCHome_TrainModelAction"] = "Start Training",
            ["LCHome_RunAnalysis"] = "Run Analysis",
            ["LCHome_RunAnalysisDescription"] = "Evaluate the trained models on independent CT scans of the lungs. The model will classify the tissue as normal, benign, or malignant.",
            ["LCHome_RunAnalysisAction"] = "New Analysis",
            ["LCHome_ReviewModels"] = "Review Models",
            ["LCHome_ReviewModelsDescription"] = "View the history of all models trained so far, analyze their final parameters, or delete those that are no longer needed.",
            ["LCHome_ReviewModelsAction"] = "Open Review",
            ["LCHome_Comparison"] = "Compare Models",
            ["LCHome_ComparisonDescription"] = "Directly compare the performance of different models across epochs (Loss, Accuracy). Select any number of models to generate graphs.",
            ["LCHome_ComparisonAction"] = "Compare Performance",
        },
        ["bs-BA"] = new()
        {
            ["LCHome_Title"] = "Analiza raka pluća",
            ["LCHome_Description"] = "Ova platforma služi za treniranje, analizu i direktnu usporedbu modela mašinskog učenja razvijenih u C# i Python programskim jezicima koristeći IQOTHNCCD CT dijagnostiku.",
            ["LCHome_TrainModel"] = "Treniraj model",
            ["LCHome_TrainModelDescription"] = "Konfigurišite i pokrenite proces treniranja novog modela. Podesite broj epoha, augmentaciju slika (rotaciju) i odaberite željeni backend sistem (C# ili Python).",
            ["LCHome_TrainModelAction"] = "Započni Treniranje",
            ["LCHome_RunAnalysis"] = "Pokreni Analizu",
            ["LCHome_RunAnalysisDescription"] = "Izvršite evaluaciju istreniranih modela na nezavisnim CT snimcima pluća. Model će klasifikovati tkivo kao normalno, benigno ili maligno.",
            ["LCHome_RunAnalysisAction"] = "Nova Analiza",
            ["LCHome_ReviewModels"] = "Pregled modela",
            ["LCHome_ReviewModelsDescription"] = "Pregledajte historiju svih do sada treniranih modela, analizirajte njihove finalne parametre ili obrišite one koji više nisu potrebni.",
            ["LCHome_ReviewModelsAction"] = "Otvori Pregled",
            ["LCHome_Comparison"] = "Uporedi Modele",
            ["LCHome_ComparisonDescription"] = "Direktno usporedite performanse različitih modela kroz epohe (Loss, Accuracy). Odaberite proizvoljan broj modela za generisanje grafikona.",
            ["LCHome_ComparisonAction"] = "Uporedi Performanse",

        }
    };

    public static string T(string key)
    {
        var currentCulture = CultureInfo.CurrentUICulture.Name;

        if (Translations.TryGetValue(currentCulture, out var localizedStrings) &&
            localizedStrings.TryGetValue(key, out var value))
        {
            return value;
        }

        if (Translations["en-US"].TryGetValue(key, out var fallbackValue))
        {
            return fallbackValue;
        }

        return key;
    }
}