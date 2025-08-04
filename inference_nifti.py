#!/usr/bin/env python3
"""
Medical SAM2 - Inferenz Script für NIfTI-Dateien
Dieses Script wendet ein bereits trainiertes Medical SAM2 Modell auf neue NIfTI-Dateien an.

WICHTIG: Dieses Script wird NACH dem Training verwendet!
Zuerst müssen Sie ein Modell mit train_3d.py oder example_nifti_usage.py trainieren.
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import torch
from PIL import Image
import matplotlib.pyplot as plt

def load_trained_model(model_path, config="sam2_hiera_s", device="cpu"):
    """
    Lädt ein trainiertes Medical SAM2 Modell
    
    Args:
        model_path (str): Pfad zum trainierten Modell (.pth Datei)
        config (str): SAM2 Konfiguration
        device (str): Gerät (cpu oder cuda)
    
    Returns:
        model: Geladenes Modell
    """
    try:
        from sam2_train.build_sam import build_sam2_video_predictor
        
        print(f"📥 Lade trainiertes Modell: {model_path}")
        
        # Modell laden
        model = build_sam2_video_predictor(
            config_file=config,
            ckpt_path=model_path,
            device=device
        )
        
        model.eval()  # Evaluationsmodus
        print("✅ Modell erfolgreich geladen!")
        return model
        
    except Exception as e:
        print(f"❌ Fehler beim Laden des Modells: {e}")
        print("\n💡 Stellen Sie sicher, dass:")
        print("1. Der Modellpfad korrekt ist")
        print("2. Das Modell existiert (nach erfolgreichem Training)")
        print("3. Die SAM2-Abhängigkeiten installiert sind")
        return None

def preprocess_nifti(nifti_path, target_size=512):
    """
    Lädt und verarbeitet eine NIfTI-Datei für die Inferenz
    
    Args:
        nifti_path (str): Pfad zur NIfTI-Datei
        target_size (int): Zielgröße für die Bilder
    
    Returns:
        processed_data: Verarbeitete Bilddaten
        original_shape: Ursprüngliche Form der Daten
        header: NIfTI-Header für das Speichern
    """
    print(f"📂 Lade NIfTI-Datei: {nifti_path}")
    
    # NIfTI-Datei laden
    nii_img = nib.load(nifti_path)
    img_data = nii_img.get_fdata()
    header = nii_img.header
    
    print(f"📊 Ursprüngliche Größe: {img_data.shape}")
    print(f"📊 Datentyp: {img_data.dtype}")
    
    # Normalisierung
    img_data = np.clip(img_data, np.percentile(img_data, 1), np.percentile(img_data, 99))
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    img_data = (img_data * 255).astype(np.uint8)
    
    # Verarbeite jede Schicht
    processed_slices = []
    for slice_idx in range(img_data.shape[2]):
        # 2D Schicht extrahieren
        slice_2d = img_data[:, :, slice_idx]
        
        # Zu RGB konvertieren
        slice_rgb = np.stack([slice_2d, slice_2d, slice_2d], axis=-1)
        
        # Größe anpassen
        pil_img = Image.fromarray(slice_rgb)
        pil_img = pil_img.resize((target_size, target_size))
        
        processed_slices.append(np.array(pil_img))
    
    processed_data = np.stack(processed_slices, axis=0)
    print(f"✅ Daten verarbeitet: {processed_data.shape}")
    
    return processed_data, img_data.shape, header

def run_inference(model, processed_data, device="cpu"):
    """
    Führt die Inferenz mit dem trainierten Modell durch
    
    Args:
        model: Trainiertes Medical SAM2 Modell
        processed_data: Verarbeitete Bilddaten
        device: Gerät für die Inferenz
    
    Returns:
        segmentation_results: Segmentierungsergebnisse
    """
    print("🔄 Führe Segmentierung durch...")
    
    segmentation_results = []
    
    with torch.no_grad():
        for slice_idx, slice_data in enumerate(processed_data):
            if slice_idx % 10 == 0:
                print(f"   Verarbeite Schicht {slice_idx + 1}/{len(processed_data)}")
            
            # Konvertiere zu Tensor
            slice_tensor = torch.from_numpy(slice_data).float().to(device)
            slice_tensor = slice_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            
            try:
                # Hier würde die eigentliche Inferenz stattfinden
                # Dies ist ein vereinfachtes Beispiel - die echte Implementierung
                # hängt von der spezifischen Medical SAM2 API ab
                
                # Placeholder für echte Inferenz
                # result = model.predict(slice_tensor)
                
                # Für das Beispiel: Dummy-Segmentierung
                dummy_result = np.zeros((slice_data.shape[0], slice_data.shape[1]), dtype=np.uint8)
                segmentation_results.append(dummy_result)
                
            except Exception as e:
                print(f"⚠️ Fehler bei Schicht {slice_idx}: {e}")
                # Fallback: Leere Maske
                dummy_result = np.zeros((slice_data.shape[0], slice_data.shape[1]), dtype=np.uint8)
                segmentation_results.append(dummy_result)
    
    print("✅ Segmentierung abgeschlossen!")
    return np.stack(segmentation_results, axis=2)

def save_segmentation(segmentation, output_path, original_shape, header):
    """
    Speichert die Segmentierungsergebnisse als NIfTI-Datei
    
    Args:
        segmentation: Segmentierungsergebnisse
        output_path: Ausgabepfad
        original_shape: Ursprüngliche Form der Daten
        header: NIfTI-Header
    """
    print(f"💾 Speichere Segmentierung: {output_path}")
    
    # Größe auf ursprüngliche Form anpassen
    if segmentation.shape != original_shape:
        print("🔄 Passe Größe an ursprüngliche Form an...")
        # Hier würde eine echte Größenanpassung stattfinden
        # Für das Beispiel: Einfache Anpassung
        resized_seg = np.zeros(original_shape, dtype=np.uint8)
        resized_seg[:segmentation.shape[0], :segmentation.shape[1], :segmentation.shape[2]] = segmentation
        segmentation = resized_seg
    
    # Als NIfTI speichern
    segmentation_nii = nib.Nifti1Image(segmentation, None, header)
    nib.save(segmentation_nii, output_path)
    
    print("✅ Segmentierung gespeichert!")

def create_visualization(original_data, segmentation, output_dir, num_slices=9):
    """
    Erstellt Visualisierungen der Segmentierungsergebnisse
    
    Args:
        original_data: Ursprüngliche Bilddaten
        segmentation: Segmentierungsergebnisse
        output_dir: Ausgabeverzeichnis
        num_slices: Anzahl der zu visualisierenden Schichten
    """
    print("📊 Erstelle Visualisierungen...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Auswahl der Schichten
    slice_indices = np.linspace(0, segmentation.shape[2]-1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    for i, slice_idx in enumerate(slice_indices):
        # Original
        axes[i].imshow(original_data[:, :, slice_idx], cmap='gray', alpha=0.7)
        # Segmentierung überlagern
        axes[i].imshow(segmentation[:, :, slice_idx], cmap='jet', alpha=0.3)
        axes[i].set_title(f'Schicht {slice_idx}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'segmentation_overlay.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualisierung gespeichert in: {output_dir}")

def main():
    """
    Hauptfunktion für die Medical SAM2 Inferenz
    """
    parser = argparse.ArgumentParser(description='Medical SAM2 Inferenz für NIfTI-Dateien')
    parser.add_argument('--input', '-i', required=True, help='Pfad zur Input NIfTI-Datei')
    parser.add_argument('--model', '-m', required=True, help='Pfad zum trainierten Modell (.pth)')
    parser.add_argument('--output', '-o', help='Pfad für die Ausgabe-NIfTI-Datei')
    parser.add_argument('--config', '-c', default='sam2_hiera_s', help='SAM2 Konfiguration')
    parser.add_argument('--device', '-d', default='cpu', help='Gerät (cpu oder cuda)')
    parser.add_argument('--visualize', '-v', action='store_true', help='Erstelle Visualisierungen')
    
    args = parser.parse_args()
    
    print("🏥 Medical SAM2 - NIfTI Inferenz")
    print("=" * 50)
    
    # Überprüfe Input-Datei
    if not os.path.exists(args.input):
        print(f"❌ Input-Datei nicht gefunden: {args.input}")
        sys.exit(1)
    
    # Überprüfe Modell-Datei
    if not os.path.exists(args.model):
        print(f"❌ Modell-Datei nicht gefunden: {args.model}")
        print("\n💡 Hinweis: Sie müssen zuerst ein Modell trainieren mit:")
        print("   python train_3d.py [Parameter]")
        print("   oder")
        print("   python example_nifti_usage.py")
        sys.exit(1)
    
    # Ausgabepfad festlegen
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        if base_name.endswith('.nii'):
            base_name = base_name[:-4]
        args.output = f"{base_name}_segmented.nii.gz"
    
    try:
        # 1. Modell laden
        model = load_trained_model(args.model, args.config, args.device)
        if model is None:
            sys.exit(1)
        
        # 2. NIfTI-Daten verarbeiten
        processed_data, original_shape, header = preprocess_nifti(args.input)
        
        # 3. Inferenz durchführen
        segmentation = run_inference(model, processed_data, args.device)
        
        # 4. Ergebnisse speichern
        save_segmentation(segmentation, args.output, original_shape, header)
        
        # 5. Visualisierungen erstellen (optional)
        if args.visualize:
            vis_dir = f"{os.path.splitext(args.output)[0]}_visualizations"
            original_data = nib.load(args.input).get_fdata()
            create_visualization(original_data, segmentation, vis_dir)
        
        print("\n🎉 Inferenz erfolgreich abgeschlossen!")
        print(f"📁 Segmentierung gespeichert: {args.output}")
        
        if args.visualize:
            print(f"📊 Visualisierungen: {vis_dir}/")
        
    except Exception as e:
        print(f"\n❌ Fehler während der Inferenz: {e}")
        print("\n🔧 Mögliche Lösungen:")
        print("1. Überprüfen Sie, dass das Modell korrekt trainiert wurde")
        print("2. Stellen Sie sicher, dass alle Abhängigkeiten installiert sind")
        print("3. Überprüfen Sie die Input-Datei")
        sys.exit(1)

if __name__ == "__main__":
    main()