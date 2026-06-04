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
1. Uvod (2-3 stranice)
•	Objasni problem: Python je standard za ML, ali enterprise firme koriste C# i .NET.
•	Cilj rada: Dokazati da li se C# može takmičiti sa Pythonom koristeći isti C++ engine (LibTorch) u pozadini.
2. Teorijska osnova (5-7 stranica)
•	Šta su Konvolutivne neuralne mreže (CNN) i kako prepoznaju slike (rak pluća)?
•	Šta je Kaggle i kakav je dataset koristen?
•	Objašnjenje metrika: Šta je Recall, šta je Precision, šta je Loss?
3. Arhitektura sistema
•	Kako je postavljen .NET Aspire da orkestrira sve?
•	Python Backend: Keras/PyTorch i FastAPI.
•	C# Backend: TorchSharp/ML.NET i ASP.NET Core API.
•	Frontend & Baza: Blazor UI i SQLite 
4. Rezultati i Uporedna 
•	Python vrijeme treniranja vs C# vrijeme treniranja.
•	Python Recall vs C# Recall.
•	Uporediš vrijeme inference-a 
5. Zaključak (1-2 stranice)
