### Useful links
Good documentation for torchsharp here
https://docs.whuanle.cn/zh/cs_pytorch

Studying the Impact of TensorFlow and PyTorch Bindings on
Machine Learning Software Quality
https://arxiv.org/pdf/2407.05466
https://github.com/asgaardlab/CmpMLBindings/blob/main/torch_bindings/py/cv/tch_lenet.py

### presentation
could be nice to put in the presentation the exact versiosn of libtorch we are using
what packages exactly libtorch 2.10

### word document
1. Uvod

    Problem statement and motivation (Python vs C# in enterprise).

    Goals of the thesis.

2. Teorijska pozadina

    Neural Networks & CNNs.

    PyTorch vs. TorchSharp (and LibTorch)

    Evaluation Metrics (Recall, Precision, TP/FP/FN, Macro/Weighted). - needs some work

3. Metodologija i Implementacija

    The Dataset (IQ-OTH/NCCD).

    Preprocesiranje slika

    Razdvajanje slika na trenining i validaciju

    CNN Architecture, Optimizer (Adam), and Loss Function (Cross-Entropy).

    Epochs and Batching

    System Architecture (.NET Aspire, Web APIs, SQLite).

    User Interface (Blazor & Multi-language support).

4. Testiranje i Kontrola

    Hardware, Seeds, and versions

5. Rezultati i Analiza

    The graphs! Training time comparison, Inference latency, Validation Recall.

6. Zaključak
