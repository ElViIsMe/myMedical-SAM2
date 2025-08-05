# Medical SAM2 - Masken Upgrade

## Übersicht

Das `example_nifti_usage.py` Script wurde erweitert, um Masken mit mehreren Labels zu unterstützen, die sich auch überschneiden können. Dies ermöglicht eine fortschrittlichere Multi-Label-Segmentierung für medizinische Bildverarbeitung.

## Neue Funktionalitäten

### 🎯 Multi-Label-Unterstützung
- **Mehrere Labels pro Pixel**: Jeder Pixel kann mehrere Labels haben
- **Überschneidende Labels**: Labels können sich überlappen
- **Flexible Label-Struktur**: Unterstützung für beliebig viele Labels

### 🔍 Automatische Validierung
- **Form-Überprüfung**: Stellt sicher, dass Bild- und Maskengrößen übereinstimmen
- **Label-Statistiken**: Zeigt detaillierte Informationen über jedes Label
- **Überschneidungs-Erkennung**: Identifiziert und quantifiziert Überlappungen

### 🎨 Erweiterte Visualisierung
- **Multi-Panel-Ansicht**: Zeigt Bild, Maske und Overlay nebeneinander
- **Farbkodierte Labels**: Verschiedene Labels werden in unterschiedlichen Farben dargestellt
- **Overlay-Darstellung**: Kombiniert Bild und Masken für bessere Analyse

### 🧪 Beispiel-Generatoren
- **Einfache Masken**: Verschiedene Labels ohne Überschneidungen
- **Überschneidende Masken**: Labels mit bewussten Überlappungen
- **Test-Daten**: Automatische Generierung von Test-Masken

## Masken-Format

### Struktur
```
Jeder Pixelwert repräsentiert ein Label:
- 0 = Hintergrund
- 1, 2, 3, ... = Verschiedene Objekte/Organe
```

### Beispiel
```python
# 3D-Maske mit mehreren Labels
mask_data[height, width, depth] = label_value

# Beispiel: Pixel (100, 150, 25) hat Label 2
mask_data[100, 150, 25] = 2

# Überschneidungen sind möglich (wird vom System erkannt)
```

## Verwendung

### 1. Grundlegende Verwendung
```python
# In example_nifti_usage.py
nifti_path = "path/to/your/medical_image.nii.gz"
mask_path = "path/to/your/mask_image.nii.gz"  # Optional
output_dir = "./data/my_nifti_data"

# Script ausführen
python example_nifti_usage.py
```

### 2. Beispiel-Masken erstellen
```python
# Interaktiv im Script
# Wählen Sie "j" wenn gefragt wird, ob Beispiel-Masken erstellt werden sollen
# Wählen Sie zwischen:
# 1. Einfache Masken (verschiedene Labels)
# 2. Überschneidende Masken (Labels überlappen sich)
```

### 3. Test-Script verwenden
```bash
# Einfacher Test ohne externe Abhängigkeiten
python3 simple_test.py

# Vollständiger Test mit NIfTI-Dateien
python3 test_mask_functionality.py
```

## Neue Funktionen im Detail

### `load_mask_file(mask_path)`
Lädt eine Masken-NIfTI-Datei und analysiert die Labels.

**Parameter:**
- `mask_path`: Pfad zur Masken-.nii oder .nii.gz Datei

**Rückgabe:**
- `mask_data`: 3D Maskendaten mit Labels
- `header`: NIfTI-Header-Informationen

### `validate_mask_data(mask_data, img_data)`
Validiert Maskendaten und zeigt detaillierte Statistiken.

**Parameter:**
- `mask_data`: 3D Maskendaten
- `img_data`: 3D Bilddaten

**Rückgabe:**
- `bool`: True wenn Masken gültig sind

### `create_example_mask_data(img_shape, output_path)`
Erstellt Beispiel-Maskendaten für Testzwecke.

**Parameter:**
- `img_shape`: Form der Bilddaten (H, W, D)
- `output_path`: Ausgabepfad für die Masken-NIfTI-Datei

### `create_overlapping_mask_example(img_shape, output_path)`
Erstellt Beispiel-Masken mit bewussten Überschneidungen.

### `visualize_nifti_slices(img_data, mask_data=None, num_slices=9)`
Erweiterte Visualisierung mit Multi-Panel-Ansicht.

## Kompatibilität

### Unterstützte Formate
- **NIfTI**: .nii, .nii.gz
- **Labels**: Integer-Werte (0, 1, 2, 3, ...)
- **Dimensionen**: 3D (Height, Width, Depth)
- **Datentypen**: uint8, uint16, float32

### Medical SAM2 Integration
Die erstellten Masken sind vollständig kompatibel mit dem Medical SAM2 Training:
- Automatische Konvertierung zu .npy-Format
- Korrekte Label-Struktur für Multi-Objekt-Training
- Unterstützung für BBox- und Click-Prompts

## Beispiele

### Beispiel 1: Einfache Multi-Label-Maske
```python
# Erstellt Masken mit 3 verschiedenen Labels
# Label 1: Zentrale Kreise
# Label 2: Rechteckige Region
# Label 3: Kleine Kreise
```

### Beispiel 2: Überschneidende Labels
```python
# Erstellt Masken mit bewussten Überschneidungen
# Label 1: Großer Kreis
# Label 2: Kleinerer Kreis (überschneidet mit Label 1)
# Label 3: Rechteck (überschneidet mit Label 1 und 2)
```

## Fehlerbehebung

### Häufige Probleme

1. **Form-Mismatch**
   ```
   ⚠️ Warnung: Bildgröße (256, 256, 64) stimmt nicht mit Maskengröße (256, 256, 32) übereinstimmen
   ```
   **Lösung**: Stellen Sie sicher, dass Bild- und Maskendaten die gleiche Form haben.

2. **Keine Labels gefunden**
   ```
   Gefundene Labels: [0]
   ```
   **Lösung**: Überprüfen Sie, ob Ihre Maskendatei tatsächlich Labels enthält.

3. **Überschneidungen erkannt**
   ```
   ⚠️ Überschneidung Label 1 & 2: 1234 Pixel
   ```
   **Hinweis**: Dies ist normal und wird vom System unterstützt.

### Debugging-Tipps

1. **Verwenden Sie die Validierung**:
   ```python
   validate_mask_data(mask_data, img_data)
   ```

2. **Testen Sie mit Beispiel-Daten**:
   ```python
   create_example_mask_data(img_shape)
   ```

3. **Überprüfen Sie die Visualisierung**:
   ```python
   visualize_nifti_slices(img_data, mask_data)
   ```

## Nächste Schritte

1. **Installation**: Stellen Sie sicher, dass alle Abhängigkeiten installiert sind
2. **Testen**: Verwenden Sie die Beispiel-Scripts
3. **Anpassen**: Passen Sie die Pfade in `example_nifti_usage.py` an
4. **Training**: Führen Sie das Medical SAM2 Training mit echten Masken durch

## Technische Details

### Label-Verarbeitung
Das System verarbeitet Labels wie folgt:
1. **Laden**: NIfTI-Datei wird geladen und in numpy-Array konvertiert
2. **Validierung**: Form und Label-Struktur werden überprüft
3. **Resizing**: Masken werden auf die Zielgröße skaliert
4. **Speicherung**: Als .npy-Dateien für Medical SAM2 vorbereitet

### Überschneidungs-Erkennung
```python
# Beispiel für Überschneidungs-Berechnung
overlap = np.sum((mask_data == label1) & (mask_data == label2))
```

### Performance-Optimierungen
- Effiziente numpy-Operationen für große Datensätze
- Speicheroptimierte Verarbeitung
- Batch-Verarbeitung für mehrere Schichten

## Support

Bei Fragen oder Problemen:
1. Überprüfen Sie die Fehlerbehebung
2. Verwenden Sie die Test-Scripts
3. Konsultieren Sie die Medical SAM2 Dokumentation