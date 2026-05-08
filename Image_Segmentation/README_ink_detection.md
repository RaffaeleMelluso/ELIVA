# Ink Detection from 3D X-ray Scans

Rileva automaticamente l'inchiostro nelle scansioni 3D a raggi X di frammenti di papiri antichi utilizzando una rete neurale convoluzionale 3D.

## Cosa Fa

- **Input**: Volume 3D (superficie + profondità) dai dati ELIVA
- **Output**: Maschera binaria dell'inchiostro in formato RLE
- **Modello**: 3D CNN con architettura encoder-decoder (uno strato rimosso dalla baseline)

## Requisiti

```bash
pip install tensorflow numpy pandas pillow scikit-image matplotlib
```

## Uso Veloce

### 1. CPU (Test Rapido)

```python
USE_GPU = False
DOWNSAMPLE = 4          # 512x512 images
NUM_TRAIN_SAMPLES = 3   # 3 campioni
```

Tempo: ~10 minuti su CPU moderna

### 2. GPU (Full Training)

```python
USE_GPU = True
DOWNSAMPLE = 1          # 2048x2048 full resolution
NUM_TRAIN_SAMPLES = 41  # Tutti i campioni
```

Tempo: ~1 ora su NVIDIA GPU

## File di Configurazione

Nel notebook, modifica:

| Variabile | CPU | GPU |
|-----------|-----|-----|
| `USE_GPU` | `False` | `True` |
| `DOWNSAMPLE` | `4` | `1` |
| `NUM_TRAIN_SAMPLES` | `3` | `len(train_patches)` |
| `EPOCHS` | `10` | `50+` |
| `BATCH_SIZE` | `1` | `4+` |

## Output

**File generato**: `submission.csv`

Formato Kaggle:
```
Id,Predicted
p_3homka84a6c1,1 50 2 100 ...
p_dw4waiq8rjzi,3 45 1 200 ...
```

## Struttura Dati

```
train/
  [sample_id]/
    ├── surface_volume/    (65 slice .tif)
    ├── inklabels.png      (etichetta inchiostro)
    ├── inklabels_rle.csv  (formato RLE)
    ├── mask.png           (maschera area valida)
    └── ir.png             (infrarosso - opzionale)

test/
  [sample_id]/
    ├── surface_volume/    (65 slice .tif)
    ├── mask.png           (maschera area valida)
    └── ir.png
```

## Note Tecniche

- **Normalizzazione**: Volume normalizzato a [0, 1]
- **Maschera 3D → 2D**: Collasso tramite max aggregation sulla profondità
- **Threshold**: 0.5 per binarizzazione predizioni
- **Metrica**: F1 Score (evaluation ufficiale)

## Troubleshooting

| Problema | Soluzione |
|----------|-----------|
| Memory error | Aumentare `DOWNSAMPLE` o ridurre `BATCH_SIZE` |
| GPU non riconosciuta | Verificare CUDA/cuDNN installation |
| TIFF non trovati | Verificare percorso `TRAIN_DIR` e `TEST_DIR` |
