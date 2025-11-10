# 🌱 Plant Disease Classification & AI Assistant

**Exercício da Disciplina de Deep Learning - UFPE (Universidade Federal de Pernambuco)**
**Disciplina:** Deep Learning  
**Instituição:** UFPE (Universidade Federal de Pernambuco)  
**Objetivo:** Aplicação prática de técnicas de Deep Learning em problemas reais
**Alunos:**
Eduardo Pinzon (ecp@cin.ufpe.br)
Yakmuri Cosme da Silva (ycs@cin.ufpe.br)


Este projeto implementa dois modelos de IA para agricultura:
1. **Classificador de Doenças por Imagem** (CNN - ResNet-9)
2. **Assistente de IA para Agricultura** (SLM - Phi-3 Fine-tuned)

---

## 📚 Módulos do Projeto

### 1. 🖼️ Image-based Disease Classifier (CNN)

Modelo de Deep Learning para identificar doenças em plantas a partir de imagens de folhas usando ResNet-9.

**Baseado em:** https://www.kaggle.com/code/atharvaingle/plant-disease-classification-resnet-99-2

#### Quick Start - CNN

```bash
# Instalar dependências
pip install -r requirements.txt

# Ver estatísticas do dataset
python main.py stats

# Treinar modelo
python main.py train

# Fazer predição
python main.py predict image.jpg
```

#### Dataset - CNN

- **38 classes** (doenças + plantas saudáveis)
- **14 tipos de plantas** (Maçã, Milho, Uva, Tomate, etc.)
- **70,295 imagens de treinamento**
- **17,572 imagens de validação**

⚠️ **O dataset de imagens não está incluído no repositório (~3GB)**

**Download:** https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

#### Características - CNN

- ✅ Arquitetura ResNet-9 (~11M parâmetros)
- ✅ 95-98% de acurácia
- ✅ Suporte Apple Silicon (MPS)
- ✅ Compatível com CUDA e CPU
- ✅ Normalização ImageNet
- ✅ One Cycle learning rate

---

### 2. 🤖 Agricultural AI Assistant (SLM - Small Language Model)

Assistente de IA baseado em Phi-3-mini fine-tuned com dataset customizado de doenças agrícolas.

#### Quick Start - SLM

```bash
# Instalar dependências específicas
pip install transformers accelerate peft trl datasets torch

# Treinar o modelo (fine-tuning)
cd slm
python plantDiseaserSmlTraining.py

# Testar o modelo treinado
python test_model.py
```

#### Dataset Customizado - SLM

- **Localização:** `slm/dataset/agricultural_diseases_dataset.json`
- **Formato:** Pares instrução-resposta para fine-tuning
- **Conteúdo:** Informações sobre doenças agrícolas, tratamentos e recomendações
- **Estrutura:**
  ```json
  {
    "instruction": "Pergunta sobre doença agrícola",
    "response": "Resposta detalhada com recomendações"
  }
  ```

✅ **O dataset SLM está incluído no repositório**

#### Características - SLM

- ✅ **Modelo Base:** microsoft/Phi-3-mini-4k-instruct
- ✅ **Fine-tuning:** LoRA (Low-Rank Adaptation) para eficiência
- ✅ **Dataset:** Customizado para doenças agrícolas brasileiras
- ✅ **Otimizado para Apple Silicon (MPS)**
- ✅ **Gradient Checkpointing** para economia de memória
- ✅ **Framework independente de teste**

#### Configuração - SLM

Parâmetros de treinamento em `slm/plantDiseaserSmlTraining.py`:

```python
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATASET_PATH = "./dataset/agricultural_diseases_dataset.json"
DATASET_SIZE = 10000  # Exemplos para treinamento
NUM_EPOCHS = 1
BATCH_SIZE = 1
LORA_R = 4  # Rank do LoRA
MAX_SEQ_LENGTH = 512
```

#### Performance - SLM

| Device | Tempo/Época | Memória |
|--------|-------------|---------|
| MPS (M1/M2/M3) | ~30-60 min | ~11-13 GB |
| CUDA (GPU) | ~20-40 min | ~10-12 GB |
| CPU | ~2-4 hours | ~8-10 GB |

#### Arquitetura - SLM

```
slm/
├── dataset/
│   └── agricultural_diseases_dataset.json  # Dataset customizado
├── plantDiseaserSmlTraining.py            # Script de treinamento
└── test_model.py                          # Framework de testes
```

**Modelo Treinado:**
- Output: `phi3-mini-doencas-agricolas-mps/`
- Formato: Adaptadores LoRA (~50MB)
- Uso: Carregamento rápido para inferência

#### Testes - SLM

Personalize as perguntas em `test_model.py`:

```python
TEST_QUESTIONS = [
    "Quais são as recomendações para tratamento de ferrugem?",
    "Quais as doenças comuns que afetam o cultivo de milho?",
    # Adicione suas perguntas aqui
]
```

---

## 🔧 Configuração do Ambiente

### Dependências - CNN

```bash
torch>=2.0.0
torchvision
numpy
pandas
matplotlib
```

### Dependências - SLM

```bash
transformers>=4.36.0
accelerate>=0.25.0
peft>=0.7.0
trl>=0.7.0
datasets>=2.14.0
torch>=2.0.0
```

---

## 📈 Performance Comparativa

### CNN (ResNet-9)

| Device | Tempo Total | Acurácia |
|--------|-------------|----------|
| CPU | 20-30 horas | 95-98% |
| MPS | **3-6 horas** | 95-98% |
| CUDA | 2-3 horas | 95-98% |

### SLM (Phi-3 Fine-tuned)

| Métrica | Valor |
|---------|-------|
| Modelo Base | Phi-3-mini-4k (3.8B parâmetros) |
| Parâmetros Treináveis | ~2M (LoRA) |
| Tempo de Treinamento | 30-60 min (MPS) |
| Tamanho do Modelo | ~50MB (adaptadores) |

---

## 🎯 Melhorias Implementadas

### CNN (v2.0)
1. ✅ Suporte Apple Silicon (MPS)
2. ✅ Normalização ImageNet (corrige explosão de gradiente)
3. ✅ Compatibilidade PyTorch 2.6+
4. ✅ Hiperparâmetros otimizados
5. ✅ CLI simplificada

### SLM (v1.0)
1. ✅ Fine-tuning com dataset customizado brasileiro
2. ✅ Otimizações de memória para MPS
3. ✅ Gradient checkpointing
4. ✅ Framework de testes independente
5. ✅ Logging estruturado
6. ✅ Compatibilidade multi-plataforma

---

### Conceitos Aplicados

#### CNN Module
- Transfer Learning
- Convolutional Neural Networks
- Data Augmentation
- Learning Rate Scheduling
- Batch Normalization

#### SLM Module
- Large Language Models
- Fine-tuning com LoRA
- Parameter-Efficient Fine-Tuning (PEFT)
- Prompt Engineering
- Quantização e Otimização de Memória

---

## 📄 Licença

MIT License

