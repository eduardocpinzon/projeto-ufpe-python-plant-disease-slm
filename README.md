# Plant Disease Classifier 

Deep Learning model to identify plant diseases from leaf images using ResNet-9.
Based on https://www.kaggle.com/code/atharvaingle/plant-disease-classification-resnet-99-2

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# View dataset stats
python main.py stats

# Train model
python main.py train

# Predict disease
python main.py predict image.jpg
```

## 📊 Dataset

- **38 classes** (diseases + healthy plants)
- **14 plant types** (Apple, Corn, Grape, Tomato, etc.)
- **70,295 training images**
- **17,572 validation images**

⚠️ **O dataset não está incluído no repositório devido ao seu tamanho (~3GB)**

**Download direto:**
- Acesse: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

## ⚡ Features

- ✅ ResNet-9 architecture (~11M parameters)
- ✅ 95-98% accuracy
- ✅ Apple Silicon (MPS) support
- ✅ CUDA & CPU compatible
- ✅ ImageNet normalization
- ✅ One Cycle learning rate

## 🔧 Configuration

Training parameters in `trainingModel/plant_disease_classification.py`:

```python
BATCH_SIZE = 32
EPOCHS = 10
MAX_LR = 0.001
WEIGHT_DECAY = 5e-4
```

## 📈 Performance

| Device | Time/Epoch | Total (10 epochs) |
|--------|------------|-------------------|
| CPU | 2-3 hours | 20-30 hours |
| MPS | 20-40 min | **3-6 hours** |
| CUDA | 10-20 min | 2-3 hours |

## 📝 Recent Improvements (v2.0)

1. ✅ Apple Silicon MPS support
2. ✅ ImageNet normalization (fixed gradient explosion)
3. ✅ PyTorch 2.6+ compatibility
4. ✅ Optimized hyperparameters
5. ✅ Simplified CLI

See [trainingModel/MELHORIAS.md](trainingModel/MELHORIAS.md) for details.

## 📄 License

MIT License

---

**Developed with ❤️ for farmers** | v2.0 - Nov 2025

