#!/usr/bin/env python3
"""
Beispiel für die Verwendung von Medical SAM2 mit NIfTI (.nii) Dateien
Dieses Script zeigt, wie Sie Ihre eigenen .nii Dateien verarbeiten können.

NEUE FUNKTIONALITÄT:
- Unterstützung für Masken mit mehreren Labels
- Labels können sich überschneiden
- Automatische Validierung der Maskendaten
- Beispiel-Masken-Generator für Tests
- Erweiterte Visualisierung mit Overlays

MASKEN-FORMAT:
- Jeder Pixelwert repräsentiert ein Label
- 0 = Hintergrund
- 1, 2, 3, ... = Verschiedene Objekte/Organe
- Mehrere Labels können sich überschneiden (Multi-Label-Segmentierung)

VERWENDUNG:
1. Setzen Sie 'nifti_path' auf Ihre Bilddatei
2. (Optional) Setzen Sie 'mask_path' auf Ihre Maskendatei
3. Führen Sie das Script aus
4. Falls keine Masken vorhanden sind, können Sie Beispiel-Masken erstellen

BEISPIEL:
python example_nifti_usage.py
"""

import os
import numpy as np
import nibabel as nib
import torch
from PIL import Image
import matplotlib.pyplot as plt

def load_nifti_file(nifti_path):
    """
    Lädt eine NIfTI-Datei und gibt die Bilddaten zurück
    
    Args:
        nifti_path (str): Pfad zur .nii oder .nii.gz Datei
    
    Returns:
        numpy.ndarray: 3D Bilddaten
        nibabel.Nifti1Header: Header-Informationen
    """
    print(f"Lade NIfTI-Datei: {nifti_path}")
    
    # NIfTI-Datei laden
    nii_img = nib.load(nifti_path)
    
    # Bilddaten extrahieren
    img_data = nii_img.get_fdata()
    
    # Header-Informationen
    header = nii_img.header
    
    print(f"Bildgröße: {img_data.shape}")
    print(f"Voxelgröße: {header.get_zooms()}")
    print(f"Datentyp: {img_data.dtype}")
    
    return img_data, header

def load_mask_file(mask_path):
    """
    Lädt eine Masken-NIfTI-Datei und gibt die Maskendaten zurück
    
    Args:
        mask_path (str): Pfad zur Masken-.nii oder .nii.gz Datei
    
    Returns:
        numpy.ndarray: 3D Maskendaten mit Labels
    """
    print(f"Lade Masken-NIfTI-Datei: {mask_path}")
    
    # NIfTI-Datei laden
    nii_img = nib.load(mask_path)
    
    # Maskendaten extrahieren
    mask_data = nii_img.get_fdata()
    
    # Header-Informationen
    header = nii_img.header
    
    print(f"Maskengröße: {mask_data.shape}")
    print(f"Voxelgröße: {header.get_zooms()}")
    print(f"Datentyp: {mask_data.dtype}")
    
    # Eindeutige Labels finden
    unique_labels = np.unique(mask_data)
    print(f"Gefundene Labels: {unique_labels}")
    print(f"Anzahl Labels (ohne Hintergrund): {len(unique_labels) - 1}")
    
    return mask_data, header

def create_example_mask_data(img_shape, output_path="example_mask.nii.gz"):
    """
    Erstellt Beispiel-Maskendaten für Testzwecke
    
    Args:
        img_shape (tuple): Form der Bilddaten (H, W, D)
        output_path (str): Ausgabepfad für die Masken-NIfTI-Datei
    
    Returns:
        str: Pfad zur erstellten Masken-Datei
    """
    print(f"Erstelle Beispiel-Maskendaten mit Form {img_shape}...")
    
    # Beispiel-Masken erstellen
    mask_data = np.zeros(img_shape, dtype=np.uint8)
    
    # Verschiedene Labels in verschiedenen Bereichen
    h, w, d = img_shape
    
    # Label 1: Zentrale Region
    center_h, center_w = h // 2, w // 2
    radius = min(h, w) // 4
    for i in range(h):
        for j in range(w):
            if (i - center_h)**2 + (j - center_w)**2 <= radius**2:
                mask_data[i, j, d//4:d//2] = 1  # Label 1 in Schichten d//4 bis d//2
    
    # Label 2: Rechteckige Region
    rect_h_start, rect_h_end = h//4, 3*h//4
    rect_w_start, rect_w_end = w//4, 3*w//4
    mask_data[rect_h_start:rect_h_end, rect_w_start:rect_w_end, d//2:3*d//4] = 2
    
    # Label 3: Kleine Kreise (überschneidend mit Label 1)
    for k in range(d//4, d//2, 5):
        for i in range(center_h-radius//2, center_h+radius//2, 10):
            for j in range(center_w-radius//2, center_w+radius//2, 10):
                if (i - center_h)**2 + (j - center_w)**2 <= (radius//3)**2:
                    mask_data[i, j, k] = 3
    
    # NIfTI-Header erstellen
    header = nib.Nifti1Header()
    header.set_data_shape(img_shape)
    header.set_zooms((1.0, 1.0, 1.0))  # Voxelgröße
    
    # NIfTI-Objekt erstellen und speichern
    nii_img = nib.Nifti1Image(mask_data, np.eye(4), header)
    nib.save(nii_img, output_path)
    
    print(f"✅ Beispiel-Maskendaten gespeichert: {output_path}")
    print(f"   Labels: {np.unique(mask_data)}")
    print(f"   Label 1: Zentrale Kreise in Schichten {d//4}-{d//2}")
    print(f"   Label 2: Rechteckige Region in Schichten {d//2}-{3*d//4}")
    print(f"   Label 3: Kleine Kreise (überschneidend) in Schichten {d//4}-{d//2}")
    
    return output_path

def create_overlapping_mask_example(img_shape, output_path="overlapping_mask_example.nii.gz"):
    """
    Erstellt ein Beispiel für Masken mit sich überschneidenden Labels
    
    Args:
        img_shape (tuple): Form der Bilddaten (H, W, D)
        output_path (str): Ausgabepfad für die Masken-NIfTI-Datei
    
    Returns:
        str: Pfad zur erstellten Masken-Datei
    """
    print(f"Erstelle Beispiel für überschneidende Masken mit Form {img_shape}...")
    
    # Beispiel-Masken mit Überschneidungen erstellen
    mask_data = np.zeros(img_shape, dtype=np.uint8)
    
    h, w, d = img_shape
    
    # Label 1: Großer Kreis
    center_h, center_w = h // 2, w // 2
    radius1 = min(h, w) // 3
    for i in range(h):
        for j in range(w):
            if (i - center_h)**2 + (j - center_w)**2 <= radius1**2:
                mask_data[i, j, d//4:d//2] = 1
    
    # Label 2: Kleinerer Kreis (überschneidet sich mit Label 1)
    radius2 = radius1 // 2
    for i in range(h):
        for j in range(w):
            if (i - center_h)**2 + (j - center_w)**2 <= radius2**2:
                mask_data[i, j, d//3:2*d//3] = 2
    
    # Label 3: Rechteck (überschneidet sich mit beiden)
    rect_h_start, rect_h_end = h//3, 2*h//3
    rect_w_start, rect_w_end = w//3, 2*w//3
    mask_data[rect_h_start:rect_h_end, rect_w_start:rect_w_end, d//4:3*d//4] = 3
    
    # NIfTI-Header erstellen
    header = nib.Nifti1Header()
    header.set_data_shape(img_shape)
    header.set_zooms((1.0, 1.0, 1.0))
    
    # NIfTI-Objekt erstellen und speichern
    nii_img = nib.Nifti1Image(mask_data, np.eye(4), header)
    nib.save(nii_img, output_path)
    
    print(f"✅ Überschneidende Masken gespeichert: {output_path}")
    print(f"   Labels: {np.unique(mask_data)}")
    print(f"   Label 1: Großer Kreis (Radius {radius1})")
    print(f"   Label 2: Kleinerer Kreis (Radius {radius2}) - überschneidet mit Label 1")
    print(f"   Label 3: Rechteck - überschneidet mit Label 1 und 2")
    
    # Überschneidungen berechnen
    overlap_1_2 = np.sum((mask_data == 1) & (mask_data == 2))
    overlap_1_3 = np.sum((mask_data == 1) & (mask_data == 3))
    overlap_2_3 = np.sum((mask_data == 2) & (mask_data == 3))
    
    print(f"   Überschneidungen:")
    print(f"     Label 1 & 2: {overlap_1_2} Pixel")
    print(f"     Label 1 & 3: {overlap_1_3} Pixel")
    print(f"     Label 2 & 3: {overlap_2_3} Pixel")
    
    return output_path

def validate_mask_data(mask_data, img_data):
    """
    Validiert Maskendaten und gibt Informationen über die Labels aus
    
    Args:
        mask_data (numpy.ndarray): 3D Maskendaten
        img_data (numpy.ndarray): 3D Bilddaten
    
    Returns:
        bool: True wenn Masken gültig sind
    """
    print("\n🔍 Validiere Maskendaten...")
    
    # Überprüfen der Form
    if mask_data.shape != img_data.shape:
        print(f"❌ Form-Mismatch: Bild {img_data.shape} vs Maske {mask_data.shape}")
        return False
    
    # Eindeutige Labels finden
    unique_labels = np.unique(mask_data)
    print(f"✅ Gefundene Labels: {unique_labels}")
    
    # Statistiken für jedes Label
    for label in unique_labels:
        if label == 0:
            continue  # Hintergrund überspringen
        label_pixels = np.sum(mask_data == label)
        label_percentage = (label_pixels / mask_data.size) * 100
        print(f"   Label {label}: {label_pixels:,} Pixel ({label_percentage:.2f}%)")
    
    # Überprüfen auf Überschneidungen
    if len(unique_labels) > 2:  # Mehr als Hintergrund + 1 Label
        print("✅ Mehrere Labels gefunden - Überschneidungen sind möglich")
        
        # Beispiel-Überschneidungen zeigen
        for i in range(len(unique_labels)):
            for j in range(i+1, len(unique_labels)):
                label1, label2 = unique_labels[i], unique_labels[j]
                if label1 == 0 or label2 == 0:
                    continue
                
                overlap = np.sum((mask_data == label1) & (mask_data == label2))
                if overlap > 0:
                    print(f"   ⚠️  Überschneidung Label {label1} & {label2}: {overlap} Pixel")
    
    print("✅ Maskendaten sind gültig")
    return True

def prepare_nifti_for_medsam2(img_data, mask_data=None, output_dir="./data/my_nifti_data", image_size=512):
    """
    Bereitet NIfTI-Daten für Medical SAM2 vor
    
    Args:
        img_data (numpy.ndarray): 3D NIfTI-Bilddaten
        mask_data (numpy.ndarray, optional): 3D NIfTI-Maskendaten mit Labels
        output_dir (str): Ausgabeverzeichnis
        image_size (int): Zielbildgröße für SAM2
    """
    print(f"Bereite Daten für Medical SAM2 vor...")
    
    # Ausgabeverzeichnisse erstellen
    image_dir = os.path.join(output_dir, 'Training', 'image', 'case_001')
    mask_dir = os.path.join(output_dir, 'Training', 'mask', 'case_001')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # Normalisierung der Intensitätswerte
    img_data = np.clip(img_data, np.percentile(img_data, 1), np.percentile(img_data, 99))
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    img_data = (img_data * 255).astype(np.uint8)
    
    print(f"Verarbeite {img_data.shape[2]} Schichten...")
    
    # Jede Schicht als separates Bild speichern
    for slice_idx in range(img_data.shape[2]):
        # 2D Schicht extrahieren
        slice_2d = img_data[:, :, slice_idx]
        
        # Zu RGB konvertieren (SAM2 erwartet 3-Kanal-Bilder)
        slice_rgb = np.stack([slice_2d, slice_2d, slice_2d], axis=-1)
        
        # Als PIL Image konvertieren und resize
        pil_img = Image.fromarray(slice_rgb)
        pil_img = pil_img.resize((image_size, image_size))
        
        # Als JPG speichern
        img_filename = os.path.join(image_dir, f'{slice_idx}.jpg')
        pil_img.save(img_filename)
        
        # Maske für diese Schicht erstellen
        if mask_data is not None and slice_idx < mask_data.shape[2]:
            # 2D Maskenschicht extrahieren
            mask_slice = mask_data[:, :, slice_idx]
            
            # Maske auf die Zielgröße resizen
            mask_pil = Image.fromarray(mask_slice.astype(np.uint8))
            mask_pil = mask_pil.resize((image_size, image_size), Image.NEAREST)
            mask_slice_resized = np.array(mask_pil)
            
            # Labels in der resized Maske anzeigen
            unique_labels = np.unique(mask_slice_resized)
            if len(unique_labels) > 1:  # Wenn es Labels gibt (nicht nur Hintergrund)
                print(f"  Schicht {slice_idx}: Labels {unique_labels}")
        else:
            # Dummy-Maske erstellen (nur wenn keine echte Maske vorhanden)
            mask_slice_resized = np.zeros((image_size, image_size), dtype=np.uint8)
            if mask_data is None:
                print(f"  Schicht {slice_idx}: Dummy-Maske (keine echte Maske geladen)")
        
        # Maske als .npy speichern
        mask_filename = os.path.join(mask_dir, f'{slice_idx}.npy')
        np.save(mask_filename, mask_slice_resized)
    
    print(f"✅ Daten vorbereitet in: {output_dir}")
    return len(range(img_data.shape[2]))

def run_medsam2_training(data_path, exp_name="NIfTI_MedSAM2"):
    """
    Startet das Medical SAM2 Training mit den vorbereiteten Daten
    
    Args:
        data_path (str): Pfad zu den vorbereiteten Daten
        exp_name (str): Name des Experiments
    """
    print(f"Starte Medical SAM2 Training...")
    
    # Training-Befehl für CPU (angepasst für deutsche Nutzer)
    cmd = f"""python train_3d.py \\
        -net sam2 \\
        -exp_name {exp_name} \\
        -sam_ckpt ./checkpoints/sam2_hiera_small.pt \\
        -sam_config sam2_hiera_s \\
        -image_size 512 \\
        -val_freq 5 \\
        -prompt bbox \\
        -prompt_freq 2 \\
        -dataset btcv \\
        -data_path {data_path} \\
        -gpu False \\
        -b 1"""
    
    print("Führen Sie folgenden Befehl aus:")
    print(cmd)
    
    return cmd

def visualize_nifti_slices(img_data, mask_data=None, num_slices=9):
    """
    Visualisiert einige Schichten der NIfTI-Datei und optional die Masken
    
    Args:
        img_data (numpy.ndarray): 3D NIfTI-Bilddaten
        mask_data (numpy.ndarray, optional): 3D NIfTI-Maskendaten
        num_slices (int): Anzahl der zu visualisierenden Schichten
    """
    if mask_data is not None:
        # Mit Masken visualisieren
        fig, axes = plt.subplots(3, 6, figsize=(20, 12))
        
        slice_indices = np.linspace(0, img_data.shape[2]-1, num_slices, dtype=int)
        
        for i, slice_idx in enumerate(slice_indices):
            # Bild
            axes[i, 0].imshow(img_data[:, :, slice_idx], cmap='gray')
            axes[i, 0].set_title(f'Bild - Schicht {slice_idx}')
            axes[i, 0].axis('off')
            
            # Maske
            if slice_idx < mask_data.shape[2]:
                mask_slice = mask_data[:, :, slice_idx]
                unique_labels = np.unique(mask_slice)
                if len(unique_labels) > 1:  # Wenn es Labels gibt
                    axes[i, 1].imshow(mask_slice, cmap='tab10', alpha=0.7)
                    axes[i, 1].set_title(f'Maske - Schicht {slice_idx}\nLabels: {unique_labels}')
                else:
                    axes[i, 1].imshow(mask_slice, cmap='gray')
                    axes[i, 1].set_title(f'Maske - Schicht {slice_idx}\n(Keine Labels)')
            else:
                axes[i, 1].imshow(np.zeros_like(img_data[:, :, slice_idx]), cmap='gray')
                axes[i, 1].set_title(f'Maske - Schicht {slice_idx}\n(Nicht verfügbar)')
            axes[i, 1].axis('off')
            
            # Overlay
            if slice_idx < mask_data.shape[2]:
                mask_slice = mask_data[:, :, slice_idx]
                overlay = img_data[:, :, slice_idx].copy()
                if len(np.unique(mask_slice)) > 1:
                    # Farbige Overlay für verschiedene Labels
                    for label in np.unique(mask_slice):
                        if label > 0:  # Nicht Hintergrund
                            mask_bool = mask_slice == label
                            overlay[mask_bool] = overlay[mask_bool] * 0.5 + 255 * 0.5
                    axes[i, 2].imshow(overlay, cmap='gray')
                    axes[i, 2].set_title(f'Overlay - Schicht {slice_idx}')
                else:
                    axes[i, 2].imshow(img_data[:, :, slice_idx], cmap='gray')
                    axes[i, 2].set_title(f'Overlay - Schicht {slice_idx}\n(Keine Labels)')
            else:
                axes[i, 2].imshow(img_data[:, :, slice_idx], cmap='gray')
                axes[i, 2].set_title(f'Overlay - Schicht {slice_idx}')
            axes[i, 2].axis('off')
            
            # Leere Spalten für bessere Darstellung
            axes[i, 3].axis('off')
            axes[i, 4].axis('off')
            axes[i, 5].axis('off')
    else:
        # Nur Bilder visualisieren
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        slice_indices = np.linspace(0, img_data.shape[2]-1, num_slices, dtype=int)
        
        for i, slice_idx in enumerate(slice_indices):
            axes[i].imshow(img_data[:, :, slice_idx], cmap='gray')
            axes[i].set_title(f'Schicht {slice_idx}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('nifti_preview.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Vorschau gespeichert als 'nifti_preview.png'")

def main():
    """
    Hauptfunktion - Beispiel für die Verwendung mit NIfTI-Dateien
    """
    print("🏥 Medical SAM2 - NIfTI Beispiel")
    print("=" * 50)
    
    # Beispiel-Pfade (passen Sie diese an Ihre Dateien an)
    nifti_path = "path/to/your/medical_image.nii.gz"  # Ihr NIfTI-Dateipfad
    mask_path = "path/to/your/mask_image.nii.gz"  # Ihre Masken-NIfTI-Datei (optional)
    output_dir = "./data/my_nifti_data"  # Ausgabeverzeichnis
    
    # Überprüfen, ob die NIfTI-Datei existiert
    if not os.path.exists(nifti_path):
        print(f"❌ NIfTI-Datei nicht gefunden: {nifti_path}")
        print("\n📋 So verwenden Sie dieses Script:")
        print("1. Ändern Sie 'nifti_path' zu Ihrer .nii oder .nii.gz Datei")
        print("2. (Optional) Ändern Sie 'mask_path' zu Ihrer Masken-.nii oder .nii.gz Datei")
        print("3. Führen Sie das Script aus: python example_nifti_usage.py")
        print("\n💡 Unterstützte Formate:")
        print("- .nii (unkomprimiert)")
        print("- .nii.gz (komprimiert)")
        print("- 3D medizinische Bilder (CT, MRI, etc.)")
        print("\n🎯 Masken-Format:")
        print("- Jeder Pixelwert repräsentiert ein Label")
        print("- 0 = Hintergrund")
        print("- 1, 2, 3, ... = Verschiedene Objekte/Organe")
        print("- Mehrere Labels können sich überschneiden")
        return
    
    try:
        # 1. NIfTI-Datei laden
        img_data, header = load_nifti_file(nifti_path)
        
        # 2. Masken-Datei laden (falls vorhanden)
        mask_data = None
        if os.path.exists(mask_path):
            print(f"\n🎯 Masken-Datei gefunden: {mask_path}")
            mask_data, mask_header = load_mask_file(mask_path)
            
            # Überprüfen, ob Bild- und Maskengrößen übereinstimmen
            if img_data.shape != mask_data.shape:
                print(f"⚠️  Warnung: Bildgröße {img_data.shape} stimmt nicht mit Maskengröße {mask_data.shape} übereinstimmen")
                print("   Die Masken werden trotzdem verarbeitet, aber es könnte zu Problemen kommen.")
        else:
            print(f"\n⚠️  Keine Masken-Datei gefunden: {mask_path}")
            
            # Option: Beispiel-Masken erstellen
            create_example = input("\n🤔 Möchten Sie Beispiel-Maskendaten erstellen? (j/n): ").lower().strip()
            if create_example in ['j', 'ja', 'y', 'yes']:
                print("\n📋 Verfügbare Beispiel-Typen:")
                print("1. Einfache Masken (verschiedene Labels)")
                print("2. Überschneidende Masken (Labels überlappen sich)")
                
                example_type = input("Welchen Typ möchten Sie erstellen? (1/2): ").strip()
                
                if example_type == "2":
                    example_mask_path = create_overlapping_mask_example(img_data.shape)
                else:
                    example_mask_path = create_example_mask_data(img_data.shape)
                
                print(f"\n🎯 Beispiel-Masken erstellt: {example_mask_path}")
                print("   Sie können diese Datei als 'mask_path' verwenden.")
                
                use_example = input("Möchten Sie diese Beispiel-Masken verwenden? (j/n): ").lower().strip()
                if use_example in ['j', 'ja', 'y', 'yes']:
                    mask_data, mask_header = load_mask_file(example_mask_path)
                else:
                    print("   Es werden Dummy-Masken (leer) erstellt.")
            else:
                print("   Es werden Dummy-Masken (leer) erstellt.")
        
        # Maskendaten validieren (falls vorhanden)
        if mask_data is not None:
            validate_mask_data(mask_data, img_data)
        
        # 3. Daten visualisieren
        print("\n📊 Erstelle Vorschau...")
        visualize_nifti_slices(img_data, mask_data)
        
        # 4. Daten für Medical SAM2 vorbereiten
        print("\n🔄 Bereite Daten vor...")
        num_slices = prepare_nifti_for_medsam2(img_data, mask_data, output_dir)
        
        # 5. Training-Befehl generieren
        print("\n🚀 Training-Befehl:")
        training_cmd = run_medsam2_training(output_dir)
        
        print(f"\n✅ Erfolgreich {num_slices} Schichten verarbeitet!")
        
        if mask_data is not None:
            print("\n📋 Nächste Schritte:")
            print("=" * 60)
            print("🎓 PHASE 1: TRAINING (mit echten Masken)")
            print("1. Überprüfen Sie die generierten Daten in:", output_dir)
            print("2. ✅ Masken wurden erfolgreich geladen und verarbeitet")
            print("3. Führen Sie das Training aus:")
            print("   " + training_cmd.replace("\\", ""))
            print("")
            print("🚀 PHASE 2: ANWENDUNG (nach erfolgreichem Training)")
            print("4. Verwenden Sie das trainierte Modell auf neuen NIfTI-Dateien:")
            print("   python inference_nifti.py -i neue_datei.nii.gz -m trainiertes_modell.pth")
            print("")
            print("💡 ZUSAMMENFASSUNG:")
            print("   - ✅ Training mit echten annotierten Daten möglich")
            print("   - Später: Automatische Segmentierung neuer Daten")
            print("   - Das Training ist eine EINMALIGE Investition!")
        else:
            print("\n📋 Nächste Schritte (WICHTIG - Dies ist nur VORBEREITUNG!):")
            print("=" * 60)
            print("🎓 PHASE 1: TRAINING (was Sie jetzt machen müssen)")
            print("1. Überprüfen Sie die generierten Daten in:", output_dir)
            print("2. ⚠️  ERSTELLEN SIE ECHTE SEGMENTIERUNGSMASKEN:")
            print("   - Die aktuellen Masken sind DUMMY-Masken (leer)")
            print("   - Sie müssen mit Tools wie 3D Slicer echte Masken erstellen")
            print("   - Jede Maske zeigt, WO sich das zu segmentierende Organ befindet")
            print("   - Tools: 3D Slicer, ITK-SNAP, ImageJ/Fiji")
            print("   - Oder laden Sie eine Masken-NIfTI-Datei mit dem Parameter 'mask_path'")
            print("3. Führen Sie das Training aus:")
            print("   " + training_cmd.replace("\\", ""))
            print("")
            print("🚀 PHASE 2: ANWENDUNG (nach erfolgreichem Training)")
            print("4. Verwenden Sie das trainierte Modell auf neuen NIfTI-Dateien:")
            print("   python inference_nifti.py -i neue_datei.nii.gz -m trainiertes_modell.pth")
            print("")
            print("💡 ZUSAMMENFASSUNG:")
            print("   - Jetzt: Training mit annotierten Daten")
            print("   - Später: Automatische Segmentierung neuer Daten")
            print("   - Das Training ist eine EINMALIGE Investition!")
        
        print("")
        print("📚 Mehr Details: Lesen Sie MEDICAL_SAM2_WORKFLOW_DEUTSCH.md")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        print("\n🔧 Mögliche Lösungen:")
        print("- Überprüfen Sie den Dateipfad")
        print("- Stellen Sie sicher, dass nibabel installiert ist: pip install nibabel")
        print("- Überprüfen Sie das NIfTI-Dateiformat")

if __name__ == "__main__":
    main()