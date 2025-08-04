# 🏥 Medical SAM2 - Korrekte Installation (CPU-Only)

## ⚡ **Schnellinstallation**

```bash
# 1. Repository klonen
git clone https://github.com/MedicineToken/Medical-SAM2.git
cd Medical-SAM2

# 2. Automatische Installation ausführen
chmod +x install_cpu.sh
./install_cpu.sh
```

## 🔧 **Manuelle Installation (bei Problemen)**

### **Schritt 1: Umgebung erstellen**
```bash
# Verwende Python 3.10 (nicht 3.12!)
conda env create -f environment_cpu.yml
conda activate medsam2_cpu
```

### **Schritt 2: PyTorch CPU installieren**
```bash
# Offizieller PyTorch CPU-Befehl (WICHTIG!)
pip install torch==2.4.0+cpu torchvision==0.19.0+cpu torchaudio==2.4.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

### **Schritt 3: Installation testen**
```bash
# PyTorch testen
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CPU-Only: {not torch.cuda.is_available()}')"

# Wichtige Pakete testen
python -c "import nibabel, monai, hydra; print('✅ Alle Pakete verfügbar')"
```

### **Schritt 4: Checkpoints herunterladen**
```bash
bash download_ckpts.sh
```

## ❌ **Häufige Fehler vermeiden**

### **NICHT verwenden:**
```bash
# ❌ FALSCH - führt zu Kompatibilitätsproblemen
pip install torch torchvision torchaudio  # Ohne CPU-Flag
conda install pytorch  # Kann CUDA-Version installieren
```

### **✅ RICHTIG verwenden:**
```bash
# ✅ KORREKT - offizielle CPU-Installation
pip install torch==2.4.0+cpu torchvision==0.19.0+cpu torchaudio==2.4.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

## 🐍 **Python-Version**

- **✅ Empfohlen:** Python 3.10.x
- **⚠️ Problematisch:** Python 3.12.x (Kompatibilitätsprobleme)
- **❌ Nicht unterstützt:** Python < 3.9

## 🔍 **Installation überprüfen**

```bash
# Umgebung aktivieren
conda activate medsam2_cpu

# Vollständiger Test
python -c "
import torch
import torchvision
import numpy as np
import nibabel as nib
import monai
from hydra import compose

print('✅ PyTorch:', torch.__version__)
print('✅ TorchVision:', torchvision.__version__)
print('✅ NumPy:', np.__version__)
print('✅ NiBabel:', nib.__version__)
print('✅ MONAI:', monai.__version__)
print('✅ CPU-Only Mode:', not torch.cuda.is_available())
print('\\n🎉 Installation erfolgreich!')
"
```

## 🚀 **Erste Schritte nach Installation**

```bash
# 1. NIfTI-Beispiel testen
python example_nifti_usage.py

# 2. Training mit eigenen Daten starten
python train_3d.py -net sam2 -exp_name Mein_Test -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 512 -data_path ./data/meine_daten -gpu False -b 1
```

## 🆘 **Bei Problemen**

1. **Umgebung neu erstellen:**
   ```bash
   conda env remove -n medsam2_cpu
   conda clean --all
   ./install_cpu.sh
   ```

2. **PyTorch neu installieren:**
   ```bash
   conda activate medsam2_cpu
   pip uninstall torch torchvision torchaudio
   pip install torch==2.4.0+cpu torchvision==0.19.0+cpu torchaudio==2.4.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
   ```

3. **Vollständige Anleitung lesen:** `ANLEITUNG_DEUTSCH.md`

---

**Wichtiger Hinweis:** Verwenden Sie immer den offiziellen PyTorch CPU-Installationsbefehl für beste Kompatibilität!