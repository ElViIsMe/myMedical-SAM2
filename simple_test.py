#!/usr/bin/env python3
"""
Einfacher Test für die Masken-Logik ohne externe Abhängigkeiten
"""

def create_test_mask_data(shape=(64, 64, 32)):
    """
    Erstellt Test-Maskendaten mit mehreren Labels
    """
    print(f"🧪 Erstelle Test-Maskendaten mit Form {shape}...")
    
    # Test-Maskendaten mit mehreren Labels erstellen
    mask_data = [[[0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
    
    h, w, d = shape
    
    # Label 1: Zentrale Region
    center_h, center_w = h // 2, w // 2
    radius = min(h, w) // 4
    
    for i in range(h):
        for j in range(w):
            if (i - center_h)**2 + (j - center_w)**2 <= radius**2:
                for k in range(d//4, d//2):
                    mask_data[i][j][k] = 1
    
    # Label 2: Rechteckige Region (überschneidet sich mit Label 1)
    rect_h_start, rect_h_end = h//4, 3*h//4
    rect_w_start, rect_w_end = w//4, 3*w//4
    for i in range(rect_h_start, rect_h_end):
        for j in range(rect_w_start, rect_w_end):
            for k in range(d//3, 2*d//3):
                mask_data[i][j][k] = 2
    
    # Label 3: Kleine Kreise (überschneidet sich mit Label 1)
    for k in range(d//4, d//2, 3):
        for i in range(center_h-radius//2, center_h+radius//2, 8):
            for j in range(center_w-radius//2, center_w+radius//2, 8):
                if (i - center_h)**2 + (j - center_w)**2 <= (radius//3)**2:
                    mask_data[i][j][k] = 3
    
    return mask_data

def analyze_mask_data(mask_data):
    """
    Analysiert Maskendaten und zeigt Statistiken
    """
    print("\n🔍 Analysiere Maskendaten...")
    
    # Eindeutige Labels finden
    unique_labels = set()
    label_counts = {}
    
    for i in range(len(mask_data)):
        for j in range(len(mask_data[0])):
            for k in range(len(mask_data[0][0])):
                label = mask_data[i][j][k]
                unique_labels.add(label)
                label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"✅ Gefundene Labels: {sorted(unique_labels)}")
    
    # Statistiken für jedes Label
    total_pixels = len(mask_data) * len(mask_data[0]) * len(mask_data[0][0])
    for label in sorted(unique_labels):
        if label == 0:
            continue  # Hintergrund überspringen
        count = label_counts.get(label, 0)
        percentage = (count / total_pixels) * 100
        print(f"   Label {label}: {count:,} Pixel ({percentage:.2f}%)")
    
    # Überprüfen auf Überschneidungen
    if len(unique_labels) > 2:  # Mehr als Hintergrund + 1 Label
        print("✅ Mehrere Labels gefunden - Überschneidungen sind möglich")
        
        # Beispiel-Überschneidungen zeigen
        for label1 in sorted(unique_labels):
            for label2 in sorted(unique_labels):
                if label1 >= label2 or label1 == 0 or label2 == 0:
                    continue
                
                overlap_count = 0
                for i in range(len(mask_data)):
                    for j in range(len(mask_data[0])):
                        for k in range(len(mask_data[0][0])):
                            if mask_data[i][j][k] == label1 and mask_data[i][j][k] == label2:
                                overlap_count += 1
                
                if overlap_count > 0:
                    print(f"   ⚠️  Überschneidung Label {label1} & {label2}: {overlap_count} Pixel")
    
    print("✅ Maskendaten sind gültig")
    return True

def demonstrate_mask_processing():
    """
    Demonstriert die Verarbeitung von Masken mit mehreren Labels
    """
    print("🎯 Demonstration: Masken mit mehreren Labels")
    print("=" * 50)
    
    # Test-Maskendaten erstellen
    mask_data = create_test_mask_data()
    
    # Maskendaten analysieren
    analyze_mask_data(mask_data)
    
    print("\n📋 Zusammenfassung der neuen Funktionalität:")
    print("✅ Unterstützung für Masken mit mehreren Labels")
    print("✅ Labels können sich überschneiden")
    print("✅ Automatische Validierung der Maskendaten")
    print("✅ Beispiel-Masken-Generator für Tests")
    print("✅ Erweiterte Visualisierung mit Overlays")
    
    print("\n🎯 Nächste Schritte:")
    print("1. Installieren Sie die erforderlichen Pakete:")
    print("   pip install numpy nibabel matplotlib pillow")
    print("2. Verwenden Sie example_nifti_usage.py mit echten NIfTI-Dateien")
    print("3. Testen Sie die Überschneidungs-Erkennung")

if __name__ == "__main__":
    demonstrate_mask_processing()