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
1.1 Objasni problem: Python je standard za ML, ali enterprise firme koriste C# i .NET.
1.2 Cilj rada: Dokazati da li se C# može takmičiti sa Pythonom koristeći isti C++ engine (LibTorch) u pozadini.      
2. Teorijska osnova
2.1 Šta su neuralne mreže? 
2.2. Šta su Konvolutivne neuralne mreže (CNN)
2.3. Skup podataka (Kaggle dataset)
2.4. Metrike evaluacije modela (Recall, Precision, Loss)
2.5. Tehnološki okvir mašinskog učenja
Objasni PyTorch, LibTorch i TorchSharp i kako oba ekosistema dijele isti C++ engine.

3. Arhitektura sistema i metodologija testiranja
3.1 Dizajn i arhitektura CNN modela, Optimizer, Weights
3.2. Orkestracija i .NET Aspire: Kratko (pola stranice) objasni da Aspire služi da istovremeno pokrene i C# i Python backend kako bi ih mogao testirati "rame uz rame" pod istim hardverskim uslovima.
3.3. Implementacija Python okruženja: Kratko spomeni FastAPI kao most preko kojeg Blazor šalje komande PyTorchu.
3.4. Implementacija .NET okruženja: C# ASP.NET Core API i TorchSharp.
3.5 Metodologija testiranja i kontrola varijabli
3.5. Praćenje rezultata (SQLite & Blazor): Kako UI korisniku prikazuje uporedne rezultate i spašava ih u bazu radi analize.

4. Rezultati i Uporedna 
4.1 Python vrijeme treniranja vs C# vrijeme treniranja.
4.2 Python Recall vs C# Recall.
4.3 Uporediš vrijeme inference-a 

5. Zaključak
