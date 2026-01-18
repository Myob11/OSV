# OSV - Obdelava Slik in Videa (Laboratorijske vaje)

Repozitorij vsebuje laboratorijske vaje pri predmetu **Obdelava Slik in Videa** (Image and Video Processing) na magistrskem študiju.

## 📁 Struktura projekta

```
laboratorijske-vaje/
├── OSV_lib.py              # Skupna knjižnica s pomožnimi funkcijami
├── vaja1/                  # Osnove nalaganja in prikaza slik
├── vaja2/                  # Histogrami in dodajanje šuma
├── vaja3/                  # Interpolacija slik
├── vaja4/                  # Obdelava 3D medicinskih slik
├── vaja5/                  # Preslikave sivinskih vrednosti
├── vaja6/                  # Geometrijske transformacije
├── vaja7/                  # Prostorsko filtriranje
├── vaja8/                  # Interaktivno ogrodje (PyQt5)
├── vaja9/                  # Poravnava slik z nadzorovanimi tockami (GUI)
├── vaja10/                 # Diskretna Fourierova transformacija 2D
├── vaja11/                 # Ocenjevanje gibanja v videu (block matching)
├── vaja12/                 # Klasifikacija MNIST s konvolucijsko mrezo (PyTorch)
├── env/                    # Python virtualno okolje
└── test/                   # Testno okolje
```

## 🎯 Vsebina vaj

### Vaja 1: Nalaganje in prikaz slik
- Branje RAW slik (binarnih formatov)
- Funkcije `load_image()` in `display_image()`
- Shranjevanje slik v RAW formatu

### Vaja 2: Histogrami in šum
- Računanje histogramov (`computeHistogram`)
- Izris histogramov in kumulativne porazdelitvene funkcije (CDF)
- Ekvilizacija histogramov
- **Dodajanje Gaussovega šuma** (`addNoise`)
- Računanje entropije slike

### Vaja 3: Interpolacija in povečevanje slik
- Interpolacija ničtega reda (nearest neighbor)
- Bilinearna interpolacija (red 1)
- Analiza vpliva interpolacije na histogram slike

### Vaja 4: 3D medicinske slike
- Nalaganje volumetričnih podatkov (`loadImage3D`)
- Planarni prerezi (`getPlanarCrossSection`)
- Planarne projekcije (`getPlanarProjection`)
- Upodabljanje MIP (maximum/minimum intensity projection)

### Vaja 5: Preslikave sivinskih vrednosti
- Linearna preslikava (`scaleImage`)
- Oknjenje (windowing) za prikaz medicinskih slik (`WindowImage`)
- Odsekoma linearna preslikava (`sectionalScaleImage`)
- Gamma korekcija (`gammaImage`)

### Vaja 6: Geometrijske transformacije
- Afine transformacije (scale, translation, rotation, shear)
- Radialne bazne funkcije (RBF)
- Thin Plate Spline (TPS) transformacije
- Forward/backward mapping

### Vaja 7: Prostorsko filtriranje
- Implementacija konvolucije
- Razširitev prostorske domene (enlarge, constant, extrapolation, reflection)
- Glajenje (Gaussov filter, mean filter, median filter)
- Robni detektorji (Sobel, Prewitt, Laplacian)
- Nelinearne operacije

### Vaja 8: Interaktivno ogrodje
- GUI aplikacija z PyQt5
- Nalaganje in prikaz slik v realnem času
- Uporaba matplotlib Canvas za vizualizacijo
- Interaktivna obdelava slik

### Vaja 9: Poravnava slik (GUI)
- PyQt5 vmesnik za nalaganje referenčne in vhodne slike
- Izbor kontrolnih točk ter afina interpolacijska/aproksimacijska poravnava
- Izračun MSE in R2 na izbranem območju, prikaz sahovnice/razlike

### Vaja 10: 2D Fourierova transformacija
- Ročna implementacija 2D DFT/IDFT (`computeDFT2`) z možnostjo konjugacije za inverz
- Analiza spektra (`analyzeDFT2`): amplituda/faza, centriranje kvadrantov, log in linearno skaliranje
- Vizualizacija amplitudnega in faznega diagrama v frekvenčni domeni

### Vaja 11: Ocenjevanje gibanja v videu
- Branje posameznih okvirjev z OpenCV (`VideoCapture`, `loadFrame`)
- Predikcija okvirja s premikom (`framePrediction`) in vizualizacija razlike
- Block matching z logaritemskim iskanjem vektorja premika za vsak blok (`blockMatching`)
- Prikaz polja vektorjev gibanja na sliki (`displayMotionField`)

### Vaja 12: Klasifikacija MNIST s CNN (PyTorch)
- Priprava `DataLoader` za train/val/test ter normalizacija MNIST
- Konvolucijska mreža z dvema conv plastema, dropoutom in dvema polno povezavnima plastema (`Net`)
- Učenje s SGD z momentom, spremljanje kriterijske funkcije in validacije
- Ovrednotenje natannčnosti in vizualizacija napovedi na testnih vzorcih

## 🛠️ Tehnologije

- **Python 3.x**
- **NumPy** - numerične operacije in matrike
- **Matplotlib** - vizualizacija slik in grafov
- **PyQt5** - grafični vmesnik (vaja 8)
- **PIL/Pillow** - dodatna podpora za slike
- **OpenCV** - delo z videom in okvirji (vaja 11)
- **PyTorch & torchvision** - nevronske mreže (vaja 12)

## 🚀 Namestitev in uporaba

### 1. Kloniranje repozitorija
```bash
git clone https://github.com/Myob11/OSV.git
cd OSV
```

### 2. Aktivacija virtualnega okolja
```powershell
# Windows PowerShell
.\env\Scripts\Activate.ps1

# Windows CMD
.\env\Scripts\activate.bat
```

### 3. Namestitev odvisnosti (če je potrebno)
```bash
pip install numpy matplotlib ipykernel jupyter pillow
```

Za vajo 8:
```bash
pip install PyQt5
```

Za vajo 11:
```bash
pip install opencv-python
```

Za vajo 12 (priporočeno po uradnih navodilih PyTorch):
```bash
pip install torch torchvision torchaudio
```

### 4. Poganjanje posamezne vaje
```bash
python vaja1/main.py
python vaja2/main.py
# ... itd.
```

### Za uporabo v Jupyter Notebook
```bash
jupyter notebook vaja2/vaja2.ipynb
```

## 📚 OSV_lib.py - Skupna knjižnica

Centralna knjižnica vseh pomožnih funkcij:

### Osnove (Vaja 1)
- `load_image(path, size, type)` - naloži RAW sliko
- `display_image(image, title, gridX, gridY, cmap)` - prikaži sliko
- `saveImage(image, path, type)` - shrani sliko v RAW format

### Histogrami (Vaja 2)
- `computeHistogram(image)` - izračunaj histogram, porazdelitev in CDF
- `displayHistogram(hist, levels, title)` - prikaži histogram
- `eqializeHistogram(image)` - ekvilizacija histograma
- `addNoise(image, std)` - dodaj Gaussov šum s standardnim odklonom

### Interpolacija (Vaja 3)
- `InterpolateImage(image, size, order)` - interpoliraj sliko (red 0 ali 1)

### 3D slike (Vaja 4)
- `loadImage3D(path, size, type)` - naloži 3D volumetrične podatke
- `getPlanarCrossSection(image, dim, normVec, loc)` - prerez v poljubni ravnini
- `getPlanarProjection(image, dim, normVec, func)` - projekcija (max, min, mean)

### Preslikave (Vaja 5)
- `scaleImage(image, k, n)` - linearna preslikava y = k*x + n
- `WindowImage(image, center, width)` - oknjenje za medicinske slike
- `sectionalScaleImage(image, inputScale, outputScale)` - odsekoma linearna preslikava
- `gammaImage(image, gamma)` - gamma korekcija

## 📝 Podatkovni formati

Slike so shranjene v **RAW formatu** (binarni podatki brez glave):
- **8-bit** slike: `np.uint8` (0-255)
- **16-bit** slike: `np.int16` ali `np.uint16`
- **3D volumni**: (višina, širina, globina) v Fortran order ('F')

Primer imena datoteke: `lena-256x512-08bit.raw`
- Širina: 256 pikslov
- Višina: 512 pikslov  
- Bitna globina: 8 bitov

## 🔬 Ključne koncepte

- **Histogram**: porazdelitev sivinskih vrednosti v sliki
- **Ekvilizacija**: izboljšanje kontrasta s prerazporeditvijo histograma
- **Interpolacija**: povečevanje/zmanjševanje slik z različnimi metodami
- **3D medicinske slike**: prerezi (sagitalni, koronalni, transverzalni) in projekcije
- **Windowing**: poudarjanje specifičnih območij Hounsfield enot (HU) pri CT slikah
- **Geometrijske transformacije**: rotacije, skaliranje, TPS za nerigidne deformacije
- **Prostorsko filtriranje**: konvolucija, glajenje, detekcija robov
- **Gaussov šum**: aditiven šum s porazdelitvijo N(0, σ²)

## 📖 Uporabni viri

- [NumPy dokumentacija](https://numpy.org/doc/)
- [Matplotlib galerija](https://matplotlib.org/stable/gallery/index.html)
- [Digital Image Processing - Gonzalez & Woods](https://www.imageprocessingplace.com/)

## 👨‍💻 Avtor

**Myob11** - Magistrski študij, 1. letnik, 1. semester

## 📄 Licenca

Ta projekt je namenjen izobraževalnim namenom.

---

*Repozitorij vsebuje implementacije nalog iz laboratorijskih vaj pri predmetu Obdelava Slik in Videa.*
