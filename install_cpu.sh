#!/bin/bash

# Medical SAM2 CPU-Only Installation Script
# Kompatibel mit Python 3.10 und offiziellen PyTorch CPU-Befehlen

echo "🏥 Medical SAM2 - CPU-Only Installation"
echo "======================================"

# Überprüfe ob conda verfügbar ist
if ! command -v conda &> /dev/null; then
    echo "❌ Conda ist nicht installiert. Bitte installieren Sie Anaconda oder Miniconda zuerst."
    echo "   Download: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "📦 Erstelle CPU-Only Conda-Umgebung..."

# Erstelle die Conda-Umgebung
conda env create -f environment_cpu.yml

if [ $? -ne 0 ]; then
    echo "❌ Fehler beim Erstellen der Conda-Umgebung."
    echo "💡 Versuchen Sie: conda clean --all && conda update conda"
    exit 1
fi

echo "✅ Conda-Umgebung erfolgreich erstellt."

# Aktiviere die Umgebung
echo "🔄 Aktiviere Umgebung..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medsam2_cpu

if [ $? -ne 0 ]; then
    echo "❌ Fehler beim Aktivieren der Umgebung."
    echo "💡 Aktivieren Sie manuell mit: conda activate medsam2_cpu"
    exit 1
fi

echo "✅ Umgebung aktiviert."

# Installiere PyTorch CPU-Version mit offiziellem Befehl
echo "🔥 Installiere PyTorch CPU-Only Version..."
pip install torch==2.4.0+cpu torchvision==0.19.0+cpu torchaudio==2.4.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

if [ $? -ne 0 ]; then
    echo "❌ Fehler bei der PyTorch Installation."
    echo "💡 Versuchen Sie manuell:"
    echo "   conda activate medsam2_cpu"
    echo "   pip install torch==2.4.0+cpu torchvision==0.19.0+cpu torchaudio==2.4.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html"
    exit 1
fi

echo "✅ PyTorch CPU-Version erfolgreich installiert."

# Überprüfe Python-Version
echo "🐍 Python-Version überprüfen..."
python_version=$(python --version 2>&1)
echo "   $python_version"

if [[ $python_version == *"3.10"* ]]; then
    echo "✅ Python 3.10 korrekt installiert."
else
    echo "⚠️  Warnung: Python 3.10 empfohlen für beste Kompatibilität."
fi

# Überprüfe PyTorch Installation
echo "🔥 PyTorch Installation überprüfen..."
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CPU verfügbar: {torch.cuda.is_available() == False}')" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ PyTorch erfolgreich installiert und getestet."
else
    echo "❌ Fehler bei der PyTorch-Überprüfung."
fi

# Installiere zusätzliche Abhängigkeiten falls nötig
echo "📋 Installiere zusätzliche Abhängigkeiten..."
pip install --upgrade pip

# Überprüfe wichtige Pakete
echo "🔍 Überprüfe wichtige Pakete..."
python -c "
import sys
packages = ['torch', 'torchvision', 'numpy', 'PIL', 'nibabel', 'monai', 'hydra']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg} - FEHLT')
        missing.append(pkg)

if missing:
    print(f'\\n⚠️  Fehlende Pakete: {missing}')
    sys.exit(1)
else:
    print('\\n✅ Alle wichtigen Pakete verfügbar!')
"

if [ $? -ne 0 ]; then
    echo "❌ Einige Pakete fehlen. Bitte überprüfen Sie die Installation."
    exit 1
fi

# SAM2 Checkpoints herunterladen
echo "📥 Lade SAM2 Checkpoints herunter..."
if [ -f "download_ckpts.sh" ]; then
    bash download_ckpts.sh
    if [ $? -eq 0 ]; then
        echo "✅ Checkpoints erfolgreich heruntergeladen."
    else
        echo "⚠️  Warnung: Fehler beim Download der Checkpoints."
        echo "💡 Laden Sie manuell herunter mit: bash download_ckpts.sh"
    fi
else
    echo "⚠️  download_ckpts.sh nicht gefunden. Checkpoints müssen manuell heruntergeladen werden."
fi

echo ""
echo "🎉 Installation abgeschlossen!"
echo "=============================="
echo ""
echo "📋 Nächste Schritte:"
echo "1. Aktivieren Sie die Umgebung: conda activate medsam2_cpu"
echo "2. Testen Sie die Installation: python example_nifti_usage.py"
echo "3. Lesen Sie die deutsche Anleitung: ANLEITUNG_DEUTSCH.md"
echo ""
echo "💻 System-Info:"
echo "   - Python: $(python --version 2>&1)"
echo "   - PyTorch: CPU-Only Version"
echo "   - Umgebung: medsam2_cpu"
echo ""
echo "🆘 Bei Problemen:"
echo "   - Überprüfen Sie Python 3.10 Kompatibilität"
echo "   - Verwenden Sie den offiziellen PyTorch CPU-Befehl"
echo "   - Lesen Sie ANLEITUNG_DEUTSCH.md für Troubleshooting"
echo ""
echo "Viel Erfolg mit Medical SAM2! 🏥"