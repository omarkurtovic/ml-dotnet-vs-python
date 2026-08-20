# Komparativna analiza ML metodologija — nacrt prezentacije
---

## 1. Naslovna slika

- Komparativna analiza ML metodologija: C# (TorchSharp) vs Python (PyTorch)
- Klasifikacija raka pluća na CT snimcima (IQ-OTH/NCCD)
- Ime, mentor, fakultet, datum

---

## 2. Uvod i motivacija

- Cilj: uporediti treniranje i inferenciju istog CV zadatka u .NET i Python okruženju
- Python/PyTorch je de facto standard u ML zajednici — ali mnogi timovi (enterprise, backend) već rade u .NET-u
- TorchSharp omogućava treniranje i inferenciju direktno u C#/.NET, bez napuštanja ekosistema
- Hipoteza rada: pošto oba frameworka koriste isti libtorch backend, **rezultati treniranja ne bi trebali suštinski da se razlikuju** — razlika je u developer experience-u, ne u modelu
- PyTorch vs TorchSharp — kratko: PyTorch = Python-native, ogroman ekosistem; TorchSharp = .NET binding nad istim libtorch jezgrom

---

## 3. Podaci

- IQ-OTH/NCCD Lung Cancer dataset — CT skenovi pluća
- 3 klase: Benign, Malignant, Normal
- Podjela train/val: **pozicijska**, ne random — prvih 75% svake klase u trening, ostatak u validaciju (identično na oba jezika, radi reproducibilnosti)
- Augmentacija: horizontalni + vertikalni flip po svakoj trening slici

---

## 4. Arhitektura CNN-a

- Ulaz: 256×256, grayscale
- Conv(1→64, k3) → ReLU → MaxPool2
- Conv(64→64, k3) → ReLU → MaxPool2
- Flatten → Linear(246016→16) → Linear(16→3)
- Identična arhitektura implementirana nezavisno u TorchSharp (C#) i PyTorch (Python)

*(slika: dijagram arhitekture)*

---

## 5. Eksperimentalne konstante

Sve što nije jezik/framework je fiksirano — ovo je kontrolna varijabla eksperimenta:

- Seed 42 (i za manual_seed i za CUDA)
- Adam optimizator, lr = 1e-4
- Batch size = 8
- Gradient clipping, max_norm = 1.0
- CrossEntropyLoss sa računatim class weights (kompenzacija neuravnoteženih klasa)
- Isti broj epoha, isti hardver (GPU), isto vrijeme mjerenja (samo trening, bez validacije)
- **6 modela ukupno: 3× C#, 3× Python** — po tri nezavisna ponavljanja radi uvida u varijansu

---

## 6. Rezultati — Vrijeme treniranja

| | Run 1 | Run 2 | Run 3 | Prosjek |
|---|---|---|---|---|
| C# | 6.10 min | 6.16 min | 5.79 min | **6.02 min (361 s)** |
| Python | 5.88 min | 5.91 min | 6.01 min | **5.93 min (356 s)** |

- Razlika: ~5 sekundi (~1.4%) — zanemarljivo, unutar varijanse između ponavljanja

*(graf: bar chart poređenja)*

---

## 7. Rezultati — Vrijeme analize slike (inferenca)

- Cold start (prva analiza nakon učitavanja modela): ~0.2 s na oba jezika — identično
- Svaka naredna analiza: praktično 0 s (model već u memoriji)
- Nema mjerljive razlike u inferenci između TorchSharp i PyTorch

---

## 8. Rezultati — Tačnost (Accuracy)

| | Run 1 | Run 2 | Run 3 | Prosjek |
|---|---|---|---|---|
| C# | 82.55% | 83.27% | 81.82% | **82.55%** |
| Python | 81.82% | 82.18% | 80.36% | **81.45%** |

- C# u prosjeku ~1.1 procentna poena bolji — unutar varijanse ponavljanja

*(graf: slika 9 iz rada)*

---

## 9. Rezultati — Makro odziv i Makro F1 mjera

| Metrika | C# prosjek | Python prosjek |
|---|---|---|
| Makro odziv (recall) | 79.22% | **79.95%** |
| Makro F1 | **79.81%** | 78.87% |

- Mješoviti rezultati — svaki jezik "pobjeđuje" na po jednoj metrici, razlika < 1pp
- Zaključak: nema sistematske prednosti nijednog frameworka

*(grafovi: slika 6 i slika 7 iz rada)*

---

## 10. Rezultati — Odziv maligne klase

- Klinički najbitnija metrika — false negative kod maligne klase je najskuplja greška

| | Run 1 | Run 2 | Run 3 | Prosjek |
|---|---|---|---|---|
| C# | 98.58% | 99.29% | 98.58% | **98.82%** |
| Python | 98.58% | 97.16% | 97.87% | **97.87%** |

- C# konzistentniji (manja varijansa) i nešto bolji prosjek

*(graf: slika 8 iz rada)*

---

## 11. Sinteza rezultata

- Sve razlike u metrikama su **< 1.5 procentna poena** — manje ili slično kao varijansa između tri ponavljanja unutar istog jezika
- Vrijeme treniranja i inferencije: praktično identično
- **Zaključak eksperimenta**: kada su arhitektura, hiperparametri, podaci i hardver identični, izbor frameworka (PyTorch vs TorchSharp) ne utiče značajno na kvalitet niti brzinu treniranja modela
- Ograničenje: rezultat vrijedi za ovaj konkretan (relativno mali) CNN i zadatak — za generalniji zaključak potrebno bi bilo testirati na više arhitektura (veći modeli, drugi tipovi podataka, transformeri)

---

## 12. PyTorch vs TorchSharp — Developer experience

- Iako su **numerički rezultati uporedivi**, cijena razvoja nije ista
- PyTorch: ogromna zajednica, opsežna dokumentacija, bezbroj tutorijala i primjera, gotovo svaki novi ML paper prvo izlazi kao PyTorch implementacija
- TorchSharp: mala zajednica, oskudna dokumentacija, malo primjera van službenih repo-a, API mjestimično manje idiomatski za .NET, teže debug-ovanje i traženje rješenja za probleme
- Praktična implikacija: izbor frameworka treba da zavisi od tima i postojeće infrastrukture, ne od očekivane razlike u performansama modela

---

## 13. Zaključak

- TorchSharp je validna opcija za .NET timove koji žele ML bez napuštanja ekosistema — rezultati su uporedivi s PyTorch-om
- Trade-off: manje trenje pri integraciji sa .NET/enterprise sistemima, ali veće trenje tokom razvoja zbog manje zrelog ekosistema i dokumentacije
- Budući rad: testirati hipotezu na drugim zadacima/arhitekturama (veći modeli, drugi tipovi podataka), veći dataset, distribuirano treniranje

---

## 14. Hvala na pažnji / Pitanja
