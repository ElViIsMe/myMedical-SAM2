# 🏥 Medical SAM2 - Deutsche Anleitung für NIfTI-Dateien

## 📋 Übersicht

Medical SAM2 ist ein fortschrittliches Segmentierungsmodell für medizinische Bilder, das **vollständig mit NIfTI (.nii) Dateien** kompatibel ist und jetzt auch auf **CPU-only Systemen** (ohne NVIDIA Grafikkarte) läuft.

### ✅ **Unterstützte Dateiformate:**
- `.nii` (unkomprimiert)
- `.nii.gz` (komprimiert)
- CT-Scans, MRT-Bilder, PET-Scans
- 2D und 3D medizinische Bilddaten

---

## 🚀 **Schritt 1: Installation**

### **Option A: CPU-Only (empfohlen für Computer ohne NVIDIA GPU)**

#### **Automatische Installation (empfohlen):**
```bash
# Repository klonen
git clone https://github.com/MedicineToken/Medical-SAM2.git
cd Medical-SAM2

# Automatisches Installationsskript ausführen
chmod +x install_cpu.sh
./install_cpu.sh
```

#### **Manuelle Installation:**
```bash
# Repository klonen
git clone https://github.com/MedicineToken/Medical-SAM2.git
cd Medical-SAM2

# CPU-Only Umgebung installieren (Python 3.10)
conda env create -f environment_cpu.yml
conda activate medsam2_cpu

# PyTorch CPU-Version mit aktuellem offiziellen Befehl installieren
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# SAM2 Checkpoints herunterladen
bash download_ckpts.sh
```

### **Option B: GPU-Version (für NVIDIA Grafikkarten)**

```bash
# Repository klonen
git clone https://github.com/MedicineToken/Medical-SAM2.git
cd Medical-SAM2

# GPU Umgebung installieren
conda env create -f environment.yml
conda activate medsam2

# SAM2 Checkpoints herunterladen
bash download_ckpts.sh
```

---

## 📁 **Schritt 2: Datenstruktur für NIfTI-Dateien**

### **Automatische Vorbereitung mit dem Beispiel-Script:**

```bash
# Beispiel-Script ausführen
python example_nifti_usage.py
```

**Oder manuell:**

Erstellen Sie folgende Ordnerstruktur:

```
data/
└── ihr_datensatz/
    ├── Training/
    │   ├── image/
    │   │   └── case_001/
    │   │       ├── 0.jpg
    │   │       ├── 1.jpg
    │   │       └── ...
    │   └── mask/
    │       └── case_001/
    │           ├── 0.npy
    │           ├── 1.npy
    │           └── ...
    └── Test/
        ├── image/
        └── mask/
```

### **NIfTI zu Medical SAM2 Format konvertieren:**

```python
import nibabel as nib
import numpy as np
from PIL import Image
import os

def convert_nifti_to_sam2_format(nifti_path, output_dir, case_name="case_001"):
    """Konvertiert NIfTI-Datei in Medical SAM2 Format"""
    
    # NIfTI laden
    nii_img = nib.load(nifti_path)
    img_data = nii_img.get_fdata()
    
    # Ordner erstellen
    image_dir = os.path.join(output_dir, 'Training', 'image', case_name)
    mask_dir = os.path.join(output_dir, 'Training', 'mask', case_name)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # Normalisierung
    img_data = np.clip(img_data, np.percentile(img_data, 1), np.percentile(img_data, 99))
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    img_data = (img_data * 255).astype(np.uint8)
    
    # Jede Schicht speichern
    for slice_idx in range(img_data.shape[2]):
        # 2D Schicht
        slice_2d = img_data[:, :, slice_idx]
        
        # RGB konvertieren
        slice_rgb = np.stack([slice_2d, slice_2d, slice_2d], axis=-1)
        
        # Als Bild speichern
        pil_img = Image.fromarray(slice_rgb)
        pil_img = pil_img.resize((512, 512))  # SAM2 Standardgröße
        pil_img.save(os.path.join(image_dir, f'{slice_idx}.jpg'))
        
        # Dummy-Maske (ersetzen Sie dies durch echte Segmentierungsmasken)
        dummy_mask = np.zeros((512, 512), dtype=np.uint8)
        np.save(os.path.join(mask_dir, f'{slice_idx}.npy'), dummy_mask)

# Verwendung:
convert_nifti_to_sam2_format('ihr_ct_scan.nii.gz', './data/mein_datensatz')
```

---

## 🎯 **Schritt 3: Training starten**

### **3D Training (für Volumen-Segmentierung):**

```bash
# CPU-Only Training
python train_3d.py \
    -net sam2 \
    -exp_name Mein_NIfTI_Experiment \
    -sam_ckpt ./checkpoints/sam2_hiera_small.pt \
    -sam_config sam2_hiera_s \
    -image_size 512 \
    -val_freq 5 \
    -prompt bbox \
    -prompt_freq 2 \
    -dataset btcv \
    -data_path ./data/mein_datensatz \
    -gpu False \
    -b 1

# GPU Training (falls verfügbar)
python train_3d.py \
    -net sam2 \
    -exp_name Mein_NIfTI_Experiment_GPU \
    -sam_ckpt ./checkpoints/sam2_hiera_small.pt \
    -sam_config sam2_hiera_s \
    -image_size 1024 \
    -val_freq 5 \
    -prompt bbox \
    -prompt_freq 2 \
    -dataset btcv \
    -data_path ./data/mein_datensatz \
    -b 2
```

### **2D Training (für Schicht-basierte Segmentierung):**

```bash
# CPU-Only Training
python train_2d.py \
    -net sam2 \
    -exp_name Mein_2D_NIfTI_Experiment \
    -vis 1 \
    -sam_ckpt ./checkpoints/sam2_hiera_small.pt \
    -sam_config sam2_hiera_s \
    -image_size 512 \
    -out_size 512 \
    -b 1 \
    -val_freq 5 \
    -dataset REFUGE \
    -data_path ./data/mein_datensatz \
    -gpu False
```

---

## 🔧 **Schritt 4: Parameter-Anpassung für verschiedene Anwendungsfälle**

### **CT-Scans (Computertomographie):**
```bash
python train_3d.py \
    -image_size 512 \
    -prompt bbox \
    -prompt_freq 3 \
    -b 1 \
    # ... weitere Parameter
```

### **MRT-Bilder (Magnetresonanztomographie):**
```bash
python train_3d.py \
    -image_size 768 \
    -prompt click \
    -prompt_freq 2 \
    -b 1 \
    # ... weitere Parameter
```

### **Kleine Strukturen (z.B. Läsionen):**
```bash
python train_2d.py \
    -image_size 1024 \
    -out_size 1024 \
    -b 1 \
    # ... weitere Parameter
```

---

## 📊 **Schritt 5: Ergebnisse und Monitoring**

### **Training überwachen:**
```bash
# TensorBoard starten
tensorboard --logdir logs/

# Im Browser öffnen: http://localhost:6006
```

### **Ergebnisse finden:**
- **Modelle:** `logs/[experiment_name]/Model/`
- **Logs:** `logs/[experiment_name]/Log/`
- **Visualisierungen:** `logs/[experiment_name]/Samples/`

---

## 💡 **Praktische Tipps für NIfTI-Dateien**

### **1. Datenvorverarbeitung:**
```python
# Intensitätsnormalisierung für CT
def normalize_ct(img_data, window_center=40, window_width=400):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img_data = np.clip(img_data, img_min, img_max)
    return (img_data - img_min) / (img_max - img_min)

# Intensitätsnormalisierung für MRT
def normalize_mri(img_data):
    return (img_data - img_data.mean()) / img_data.std()
```

### **2. Segmentierungsmasken erstellen:**
```python
# Beispiel für binäre Segmentierung
def create_binary_mask(mask_data, target_label=1):
    binary_mask = (mask_data == target_label).astype(np.uint8)
    return binary_mask

# Beispiel für Multi-Class Segmentierung
def create_multiclass_mask(mask_data, label_mapping):
    output_mask = np.zeros_like(mask_data)
    for old_label, new_label in label_mapping.items():
        output_mask[mask_data == old_label] = new_label
    return output_mask
```

### **3. Datenaugmentation:**
```python
# Rotation und Flip für medizinische Bilder
def augment_medical_data(img_data, mask_data):
    # Zufällige Rotation
    angle = np.random.uniform(-15, 15)
    img_rotated = rotate(img_data, angle, reshape=False)
    mask_rotated = rotate(mask_data, angle, reshape=False)
    
    # Zufälliger Flip
    if np.random.random() > 0.5:
        img_rotated = np.fliplr(img_rotated)
        mask_rotated = np.fliplr(mask_rotated)
    
    return img_rotated, mask_rotated
```

---

## 🎯 **Anwendungsbeispiele**

### **1. Leber-Segmentierung in CT-Scans:**
```bash
python train_3d.py \
    -exp_name Leber_Segmentierung \
    -image_size 512 \
    -prompt bbox \
    -data_path ./data/leber_ct_daten \
    -gpu False
```

### **2. Gehirntumor-Erkennung in MRT:**
```bash
python train_3d.py \
    -exp_name Gehirntumor_MRT \
    -image_size 768 \
    -prompt click \
    -data_path ./data/gehirn_mrt_daten \
    -gpu False
```

### **3. Lungen-Segmentierung:**
```bash
python train_3d.py \
    -exp_name Lungen_Segmentierung \
    -image_size 512 \
    -prompt bbox \
    -data_path ./data/lungen_ct_daten \
    -gpu False
```

---

## ⚠️ **Häufige Probleme und Lösungen**

### **Problem: Python 3.12 Kompatibilitätsprobleme**
**Lösung:**
```bash
# Verwenden Sie Python 3.10 für beste Kompatibilität
conda env remove -n medsam2_cpu
conda env create -f environment_cpu.yml  # Verwendet jetzt Python 3.10
conda activate medsam2_cpu
```

### **Problem: PyTorch Installation schlägt fehl**
**Lösung:**
```bash
# Verwenden Sie den aktuellen PyTorch CPU-Befehl
conda activate medsam2_cpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Überprüfen Sie die Installation
python -c "import torch; print(torch.__version__)"
```

### **Problem: "CUDA out of memory"**
**Lösung:**
```bash
# Kleinere Batch-Größe verwenden
-b 1

# Kleinere Bildgröße verwenden
-image_size 256

# CPU-Only Modus verwenden
-gpu False
```

### **Problem: "ModuleNotFoundError" bei wichtigen Paketen**
**Lösung:**
```bash
# Fehlende Pakete nachinstallieren
conda activate medsam2_cpu
pip install nibabel monai hydra-core omegaconf

# Alle Abhängigkeiten überprüfen
python -c "
import torch, torchvision, numpy, PIL, nibabel, monai
print('✅ Alle wichtigen Pakete verfügbar!')
"
```

### **Problem: "NIfTI-Datei kann nicht geladen werden"**
**Lösung:**
```python
# Überprüfen Sie das Dateiformat
import nibabel as nib
try:
    img = nib.load('ihre_datei.nii.gz')
    print("Datei erfolgreich geladen")
    print(f"Shape: {img.shape}, Dtype: {img.get_fdata().dtype}")
except Exception as e:
    print(f"Fehler: {e}")
    # Versuchen Sie verschiedene Dateierweiterungen
    # .nii, .nii.gz, .hdr/.img
```

### **Problem: Conda-Umgebung kann nicht erstellt werden**
**Lösung:**
```bash
# Conda aktualisieren und Cache leeren
conda update conda
conda clean --all

# Umgebung mit spezifischer Python-Version erstellen
conda create -n medsam2_cpu python=3.10
conda activate medsam2_cpu
pip install -r requirements_cpu.txt  # Falls verfügbar
```

### **Problem: "Schlechte Segmentierungsergebnisse"**
**Lösungen:**
1. **Mehr Trainingsdaten:** Mindestens 50-100 annotierte Fälle
2. **Bessere Vorverarbeitung:** Normalisierung anpassen
3. **Hyperparameter-Tuning:** Learning Rate, Batch Size anpassen
4. **Transfer Learning:** Von vortrainierten Modellen starten
5. **Python 3.10:** Verwenden Sie die empfohlene Python-Version

---

## 📈 **Performance-Optimierung**

### **CPU-Optimierung:**
```bash
# Anzahl der CPU-Kerne nutzen
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Speicher-effizientes Training
python train_3d.py -b 1 -image_size 256
```

### **Speicher-Optimierung:**
```python
# Gradient Checkpointing aktivieren
torch.utils.checkpoint.checkpoint_sequential()

# Mixed Precision für GPU
torch.cuda.amp.autocast()
```

---

## 🆘 **Support und Community**

- **GitHub Issues:** [Medical SAM2 Issues](https://github.com/MedicineToken/Medical-SAM2/issues)
- **Dokumentation:** Siehe README.md im Repository
- **Beispiele:** Verwenden Sie `example_nifti_usage.py`

---

## 📝 **Lizenz und Zitation**

Wenn Sie Medical SAM2 in Ihrer Forschung verwenden, zitieren Sie bitte:

```bibtex
@misc{zhu2024medical,
    title={Medical SAM 2: Segment medical images as video via Segment Anything Model 2},
    author={Jiayuan Zhu and Abdullah Hamdi and Yunli Qi and Yueming Jin and Junde Wu},
    year={2024},
    eprint={2408.00874},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

---

## ✅ **Zusammenfassung**

Medical SAM2 ist **vollständig kompatibel** mit NIfTI-Dateien und bietet:

- ✅ **CPU-Only Support** (keine NVIDIA GPU erforderlich)
- ✅ **Automatische NIfTI-Verarbeitung** mit nibabel
- ✅ **2D und 3D Segmentierung**
- ✅ **Deutsche Anleitung und Support**
- ✅ **Beispiel-Scripts** für einfachen Einstieg

**Viel Erfolg bei Ihren medizinischen Segmentierungsprojekten! 🏥**