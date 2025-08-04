#!/usr/bin/env python3
"""
Beispiel für die Verwendung von Medical SAM2 mit NIfTI (.nii) Dateien
Dieses Script zeigt, wie Sie Ihre eigenen .nii Dateien verarbeiten können.
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

def prepare_nifti_for_medsam2(img_data, output_dir, image_size=512):
    """
    Bereitet NIfTI-Daten für Medical SAM2 vor
    
    Args:
        img_data (numpy.ndarray): 3D NIfTI-Bilddaten
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
        
        # Dummy-Maske erstellen (für das Beispiel - in der Praxis würden Sie echte Masken haben)
        dummy_mask = np.zeros((image_size, image_size), dtype=np.uint8)
        mask_filename = os.path.join(mask_dir, f'{slice_idx}.npy')
        np.save(mask_filename, dummy_mask)
    
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

def visualize_nifti_slices(img_data, num_slices=9):
    """
    Visualisiert einige Schichten der NIfTI-Datei
    
    Args:
        img_data (numpy.ndarray): 3D NIfTI-Bilddaten
        num_slices (int): Anzahl der zu visualisierenden Schichten
    """
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
    output_dir = "./data/my_nifti_data"  # Ausgabeverzeichnis
    
    # Überprüfen, ob die NIfTI-Datei existiert
    if not os.path.exists(nifti_path):
        print(f"❌ NIfTI-Datei nicht gefunden: {nifti_path}")
        print("\n📋 So verwenden Sie dieses Script:")
        print("1. Ändern Sie 'nifti_path' zu Ihrer .nii oder .nii.gz Datei")
        print("2. Führen Sie das Script aus: python example_nifti_usage.py")
        print("\n💡 Unterstützte Formate:")
        print("- .nii (unkomprimiert)")
        print("- .nii.gz (komprimiert)")
        print("- 3D medizinische Bilder (CT, MRI, etc.)")
        return
    
    try:
        # 1. NIfTI-Datei laden
        img_data, header = load_nifti_file(nifti_path)
        
        # 2. Daten visualisieren
        print("\n📊 Erstelle Vorschau...")
        visualize_nifti_slices(img_data)
        
        # 3. Daten für Medical SAM2 vorbereiten
        print("\n🔄 Bereite Daten vor...")
        num_slices = prepare_nifti_for_medsam2(img_data, output_dir)
        
        # 4. Training-Befehl generieren
        print("\n🚀 Training-Befehl:")
        training_cmd = run_medsam2_training(output_dir)
        
        print(f"\n✅ Erfolgreich {num_slices} Schichten verarbeitet!")
        print("\n📋 Nächste Schritte:")
        print("1. Überprüfen Sie die generierten Daten in:", output_dir)
        print("2. Erstellen Sie echte Segmentierungsmasken für Ihre Daten")
        print("3. Führen Sie den oben gezeigten Training-Befehl aus")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        print("\n🔧 Mögliche Lösungen:")
        print("- Überprüfen Sie den Dateipfad")
        print("- Stellen Sie sicher, dass nibabel installiert ist: pip install nibabel")
        print("- Überprüfen Sie das NIfTI-Dateiformat")

if __name__ == "__main__":
    main()