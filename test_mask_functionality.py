#!/usr/bin/env python3
"""
Test-Script für die neue Masken-Funktionalität
Demonstriert das Laden und Verarbeiten von Masken mit mehreren Labels
"""

import numpy as np
import nibabel as nib
import os

def create_test_data():
    """Erstellt Test-Bild- und Maskendaten"""
    print("🧪 Erstelle Test-Daten...")
    
    # Test-Bilddaten erstellen
    img_shape = (64, 64, 32)  # Kleine Größe für Tests
    img_data = np.random.randint(0, 255, img_shape, dtype=np.uint8)
    
    # Test-Maskendaten mit mehreren Labels erstellen
    mask_data = np.zeros(img_shape, dtype=np.uint8)
    
    # Label 1: Zentrale Region
    center_h, center_w = img_shape[0] // 2, img_shape[1] // 2
    radius = min(img_shape[0], img_shape[1]) // 4
    
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if (i - center_h)**2 + (j - center_w)**2 <= radius**2:
                mask_data[i, j, img_shape[2]//4:img_shape[2]//2] = 1
    
    # Label 2: Rechteckige Region (überschneidet sich mit Label 1)
    rect_h_start, rect_h_end = img_shape[0]//4, 3*img_shape[0]//4
    rect_w_start, rect_w_end = img_shape[1]//4, 3*img_shape[1]//4
    mask_data[rect_h_start:rect_h_end, rect_w_start:rect_w_end, img_shape[2]//3:2*img_shape[2]//3] = 2
    
    # Label 3: Kleine Kreise (überschneidet sich mit Label 1)
    for k in range(img_shape[2]//4, img_shape[2]//2, 3):
        for i in range(center_h-radius//2, center_h+radius//2, 8):
            for j in range(center_w-radius//2, center_w+radius//2, 8):
                if (i - center_h)**2 + (j - center_w)**2 <= (radius//3)**2:
                    mask_data[i, j, k] = 3
    
    # NIfTI-Dateien speichern
    # Bild
    img_header = nib.Nifti1Header()
    img_header.set_data_shape(img_shape)
    img_header.set_zooms((1.0, 1.0, 1.0))
    img_nii = nib.Nifti1Image(img_data, np.eye(4), img_header)
    nib.save(img_nii, "test_image.nii.gz")
    
    # Maske
    mask_header = nib.Nifti1Header()
    mask_header.set_data_shape(img_shape)
    mask_header.set_zooms((1.0, 1.0, 1.0))
    mask_nii = nib.Nifti1Image(mask_data, np.eye(4), mask_header)
    nib.save(mask_nii, "test_mask.nii.gz")
    
    print("✅ Test-Daten erstellt:")
    print(f"   Bild: test_image.nii.gz ({img_shape})")
    print(f"   Maske: test_mask.nii.gz")
    print(f"   Labels: {np.unique(mask_data)}")
    
    # Überschneidungen berechnen
    overlap_1_2 = np.sum((mask_data == 1) & (mask_data == 2))
    overlap_1_3 = np.sum((mask_data == 1) & (mask_data == 3))
    overlap_2_3 = np.sum((mask_data == 2) & (mask_data == 3))
    
    print(f"   Überschneidungen:")
    print(f"     Label 1 & 2: {overlap_1_2} Pixel")
    print(f"     Label 1 & 3: {overlap_1_3} Pixel")
    print(f"     Label 2 & 3: {overlap_2_3} Pixel")
    
    return "test_image.nii.gz", "test_mask.nii.gz"

def test_mask_loading():
    """Testet das Laden der Masken"""
    print("\n🔍 Teste Masken-Laden...")
    
    # Test-Daten erstellen
    img_path, mask_path = create_test_data()
    
    # Masken laden
    nii_img = nib.load(mask_path)
    mask_data = nii_img.get_fdata()
    
    print(f"✅ Masken erfolgreich geladen:")
    print(f"   Form: {mask_data.shape}")
    print(f"   Datentyp: {mask_data.dtype}")
    print(f"   Labels: {np.unique(mask_data)}")
    
    # Label-Statistiken
    for label in np.unique(mask_data):
        if label == 0:
            continue
        label_pixels = np.sum(mask_data == label)
        label_percentage = (label_pixels / mask_data.size) * 100
        print(f"   Label {label}: {label_pixels:,} Pixel ({label_percentage:.2f}%)")
    
    return img_path, mask_path

def test_example_nifti_usage():
    """Testet die Integration mit example_nifti_usage.py"""
    print("\n🚀 Teste Integration mit example_nifti_usage.py...")
    
    # Test-Daten erstellen
    img_path, mask_path = test_mask_loading()
    
    print(f"\n📋 So testen Sie die neue Funktionalität:")
    print("1. Öffnen Sie example_nifti_usage.py")
    print("2. Ändern Sie die Pfade:")
    print(f"   nifti_path = '{img_path}'")
    print(f"   mask_path = '{mask_path}'")
    print("3. Führen Sie aus: python example_nifti_usage.py")
    print("\n💡 Das Script wird:")
    print("   - Die Masken mit mehreren Labels laden")
    print("   - Überschneidungen erkennen und anzeigen")
    print("   - Eine erweiterte Visualisierung erstellen")
    print("   - Die Daten für Medical SAM2 vorbereiten")

if __name__ == "__main__":
    print("🧪 Test-Script für Masken mit mehreren Labels")
    print("=" * 50)
    
    test_example_nifti_usage()
    
    print("\n✅ Test abgeschlossen!")
    print("📁 Erstellte Dateien:")
    print("   - test_image.nii.gz (Test-Bilddaten)")
    print("   - test_mask.nii.gz (Test-Maskendaten mit mehreren Labels)")
    print("\n🎯 Nächste Schritte:")
    print("   - Verwenden Sie diese Dateien mit example_nifti_usage.py")
    print("   - Experimentieren Sie mit verschiedenen Label-Kombinationen")
    print("   - Testen Sie die Überschneidungs-Erkennung")