"""
Fine-tuning de Small Language Model (Phi-3) para Classificação de Doenças Agrícolas
Otimizado para Apple Silicon (MPS)
"""

import os
# Permite uso de mais memória do MPS
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import warnings
import logging
from pathlib import Path
from typing import Dict, Any
import gc

# ===================================================================
# CONFIGURAÇÃO DE LOGGING
# MELHORIA: Logging estruturado para melhor rastreamento do treinamento
# ===================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Silenciar avisos menos importantes
warnings.filterwarnings("ignore", category=UserWarning)

# ===================================================================
# 1. DETECÇÃO E CONFIGURAÇÃO DO DEVICE
# MELHORIA: Validação robusta com fallback automático
# ===================================================================
def setup_device() -> str:
    """
    Detecta e configura o dispositivo de computação ideal.

    Returns:
        str: Device string ('mps', 'cuda', ou 'cpu')
    """
    if torch.backends.mps.is_available():
        # MELHORIA: Verificação adicional de que o MPS está funcionalmente disponível
        try:
            # Teste rápido para garantir que o MPS está funcionando
            test_tensor = torch.zeros(1, device='mps')
            del test_tensor
            logger.info(" MPS disponível e funcional (Apple Silicon)")
            return "mps"
        except Exception as e:
            logger.warning(f" MPS disponível mas com problemas: {e}")
            logger.info("Usando CPU como fallback")
            return "cpu"
    elif torch.cuda.is_available():
        logger.info(" CUDA disponível (GPU NVIDIA)")
        return "cuda"
    else:
        logger.info(" Usando CPU (sem aceleração)")
        return "cpu"

device = setup_device()

# ===================================================================
# 2. CONFIGURAÇÕES CENTRALIZADAS
# ===================================================================
class TrainingConfig:
    """
    Configuração centralizada do treinamento.
    """
    # Modelo e Dataset
    MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
    DATASET_PATH = "./dataset/agricultural_diseases_dataset.json"
    DATASET_SIZE = 10000
    NEW_MODEL_NAME = "phi3-mini-doencas-agricolas-mps"

    # Hiperparâmetros de Treinamento
    NUM_EPOCHS = 1
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 8  # CORREÇÃO: Aumentado de 4 para 8
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.001
    MAX_SEQ_LENGTH = 512  # CORREÇÃO: Reduzido de 1024 para 512

    # Configuração LoRA
    LORA_R = 4  # CORREÇÃO: Reduzido de 8 para 4 (menos parâmetros)
    LORA_ALPHA = 8  # CORREÇÃO: Ajustado proporcionalmente
    LORA_DROPOUT = 0.1

    # Inferência
    MAX_NEW_TOKENS = 150
    TEMPERATURE = 0.7
    TOP_P = 0.9

config = TrainingConfig()
logger.info(f" Configuração carregada: {config.MODEL_NAME}")


# ===================================================================
# 3. CARREGAMENTO E PREPARAÇÃO DO DATASET
# MELHORIA: Validação de dados e formatação robusta
# ===================================================================
def load_and_prepare_dataset(dataset_path: str, size: int = None) -> Any:
    """
    Carrega e prepara o dataset local para treinamento.

    Args:
        dataset_path: Caminho relativo para o arquivo JSON
        size: Número de exemplos a carregar (None para todos)

    Returns:
        Dataset formatado e pronto para treinamento
    """
    logger.info(f" Carregando dataset local: {dataset_path}")

    try:
        # Converte para Path e valida existência
        dataset_file = Path(dataset_path)

        if not dataset_file.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado: {dataset_file.absolute()}\n"
                f"Certifique-se de que o arquivo existe no diretório do projeto."
            )

        # Carrega dataset JSON local
        dataset = load_dataset('json', data_files=str(dataset_file), split='train')

        logger.info(f" Dataset carregado: {len(dataset)} exemplos")

        # Limita tamanho se especificado
        if size is not None and size < len(dataset):
            dataset = dataset.select(range(size))
            logger.info(f" Dataset limitado a {size} exemplos")

        # Validação básica
        if len(dataset) == 0:
            raise ValueError("Dataset vazio!")

        return dataset
    except Exception as e:
        logger.error(f" Erro ao carregar dataset: {e}")
        raise

def formatar_prompt(exemplo: Dict[str, Any]) -> Dict[str, str]:
    """
    Formata um exemplo do dataset no formato esperado pelo Phi-3.

    MELHORIA: Validação de campos obrigatórios

    Args:
        exemplo: Dicionário com 'instruction' e 'response'

    Returns:
        Dicionário com campo 'text' formatado
    """
    # Validação de campos
    if 'instruction' not in exemplo or 'response' not in exemplo:
        logger.warning("Exemplo sem 'instruction' ou 'response', pulando...")
        return {"text": ""}

    # Formatação no estilo Phi-3 chat
    texto_formatado = (
        f"<|user|>\n{exemplo['instruction']}<|end|>\n"
        f"<|assistant|>\n{exemplo['response']}<|end|>"
    )

    return {"text": texto_formatado}

# Carrega e prepara o dataset
logger.info(" Preparando dataset...")
dataset = load_and_prepare_dataset(
    config.DATASET_PATH,
    config.DATASET_SIZE
)

# Aplica formatação
dataset = dataset.map(formatar_prompt)
logger.info(" Dataset formatado e pronto")

# ===================================================================
# 4. CARREGAMENTO DO MODELO E TOKENIZADOR
# MELHORIA: Otimizado para MPS com gestão de memória aprimorada
# ===================================================================
def load_model_and_tokenizer(model_name: str, device: str) -> tuple:
    """
    Carrega modelo e tokenizador com configurações otimizadas.

    MELHORIAS: Máxima economia de memória
    - Gradient checkpointing ativado
    - Limite de memória configurado
    - Float16 para economia

    Args:
        model_name: Nome do modelo no HuggingFace
        device: Device de computação ('mps', 'cuda', 'cpu')

    Returns:
        Tupla (model, tokenizer)
    """
    logger.info(f" Carregando modelo: {model_name}")

    try:
        # CORREÇÃO: Carregamento com máxima economia de memória
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            max_memory={0: "16GB"},  # CORREÇÃO: Limita memória do modelo
            attn_implementation="eager"
        )

        # CORREÇÃO: Desabilita cache e ativa gradient checkpointing
        #model.config.use_cache = False
        model.gradient_checkpointing_enable()  # Troca memória por velocidade

        logger.info(" Modelo carregado com economia de memória")

        # Carrega tokenizador
        logger.info(" Carregando tokenizador...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
            model_max_length=config.MAX_SEQ_LENGTH  # CORREÇÃO: Limita tamanho
        )

        #  Configuração do tokenizador
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(" pad_token configurado como eos_token")

        tokenizer.padding_side = "right"

        logger.info(" Tokenizador configurado")

        return model, tokenizer

    except Exception as e:
        logger.error(f" Erro ao carregar modelo/tokenizador: {e}")
        raise

# Carrega modelo e tokenizador
model, tokenizer = load_model_and_tokenizer(config.MODEL_NAME, device)

# ===================================================================
# 5. CONFIGURAÇÃO DO LoRA
# MELHORIA: Documentação detalhada e configuração otimizada
# ===================================================================
def create_lora_config() -> LoraConfig:
    """
    Cria configuração LoRA otimizada para fine-tuning eficiente.

    LoRA (Low-Rank Adaptation) permite fine-tuning com poucos parâmetros treináveis:
    - r=8: Rank das matrizes LoRA (balanço entre qualidade e eficiência)
    - lora_alpha=16: Fator de escala (tipicamente 2x o rank)
    - lora_dropout=0.1: Regularização para evitar overfitting
    - target_modules: Camadas de atenção do Phi-3 a adaptar

    CORREÇÃO:
    - "auto" não funciona com Phi-3, especificamos manualmente:
      - "q_proj", "k_proj", "v_proj": Projeções de query, key, value (atenção)
      - "o_proj": Projeção de saída da atenção
      - "gate_proj", "up_proj", "down_proj": Camadas MLP (opcional, comentadas para economia)

    Returns:
        LoraConfig configurado
    """
    logger.info("⚙ Configurando LoRA...")

    # CORREÇÃO: Módulos específicos do Phi-3
    # Focando nas camadas de atenção para melhor eficiência
    target_modules = [
        "q_proj",      # Query projection
        "k_proj",      # Key projection
        "v_proj",      # Value projection
        "o_proj",      # Output projection
        # Opcional: adicione camadas MLP para maior capacidade (usa mais memória)
        # "gate_proj",
        # "up_proj",
        # "down_proj",
    ]

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,  # CORREÇÃO: Lista explícita ao invés de "auto"
    )

    logger.info(f" LoRA configurado: r={config.LORA_R}, alpha={config.LORA_ALPHA}")
    logger.info(f"   Target modules: {', '.join(target_modules)}")
    return lora_config

peft_config = create_lora_config()

# ===================================================================
# 6. CONFIGURAÇÃO DO TREINAMENTO
# MELHORIA: Otimizado para MPS com documentação completa
# ===================================================================
def create_training_arguments(output_dir: str, device: str) -> TrainingArguments:
    """
    Cria argumentos de treinamento otimizados para Apple Silicon (MPS).

    MELHORIAS IMPLEMENTADAS:
    - Otimizador PyTorch nativo (compatível com MPS)
    - FP16 para economia de memória e velocidade
    - Gradient accumulation para simular batches maiores
    - Logging estruturado para monitoramento
    - Configurações específicas para MPS

    Args:
        output_dir: Diretório para salvar checkpoints
        device: Device de computação

    Returns:
        TrainingArguments configurado
    """
    logger.info(" Configurando argumentos de treinamento...")

    # MELHORIA: Detecta se está usando MPS
    use_mps = (device == "mps")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,

        # MELHORIA: Otimizador compatível com MPS
        # 'adamw_torch' usa implementação PyTorch nativa (não BitsAndBytes)
        optim="adamw_torch",

        # Estratégia de salvamento e logging
        save_strategy="epoch",
        logging_steps=25,
        logging_dir=f"{output_dir}/logs",  # MELHORIA: Diretório específico para logs

        # Hiperparâmetros de otimização
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=0.3,  # Gradient clipping
        warmup_ratio=0.03,  # 3% dos steps para warmup

        # MELHORIA: Precisão mista otimizada para MPS
        # FP16 é suportado pelo MPS, BF16 não
        fp16=True if use_mps else False,
        bf16=False,

        # Otimizações adicionais
        max_steps=-1,  # Usa num_train_epochs ao invés de steps fixos
        group_by_length=True,  # Agrupa exemplos de tamanho similar
        lr_scheduler_type="constant",  # Learning rate constante

        # MELHORIA: Configuração específica para MPS
        use_mps_device=use_mps,

        # MELHORIA: Desabilita métricas desnecessárias para economizar memória
        report_to="none",  # Desabilita wandb, tensorboard, etc.

        # CORREÇÃO: Otimizações de memória
        gradient_checkpointing=True,  # Reduz memória
        eval_strategy="no",  # Sem validação para economizar memória
        save_total_limit=1,  # Mantém apenas 1 checkpoint
    )

    logger.info(f" Training args configurado: {config.NUM_EPOCHS} epochs, batch={config.BATCH_SIZE}")
    logger.info(f"   Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"   Gradient checkpointing: ATIVADO")
    return args

training_arguments = create_training_arguments(
    f"./{config.NEW_MODEL_NAME}",
    device
)

# ===================================================================
# 7. INICIALIZAÇÃO DO TRAINER E TREINAMENTO
# MELHORIA: Tratamento de erros e monitoramento aprimorado
# ===================================================================
def train_model(model, dataset, peft_config, tokenizer, training_args, output_dir: str):
    """
    Inicializa o trainer e executa o treinamento com monitoramento.

    CORREÇÃO:
    - Removido 'max_seq_length' (não suportado)
    - Usa 'processing_class' ao invés de 'tokenizer'
    - Limpeza de memória periódica
    """
    logger.info(" Inicializando SFTTrainer...")

    # Limpa memória antes de treinar
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    try:
        # CORREÇÃO: Removido max_seq_length
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
            args=training_args,
        )
        logger.info(" Trainer inicializado com sucesso")

        # Treinamento
        logger.info(" Iniciando fine-tuning...")
        logger.info(f"   Device: {device}")
        logger.info(f"   Dataset: {len(dataset)} exemplos")
        logger.info(f"   Epochs: {config.NUM_EPOCHS}")
        logger.info(f"   Batch size: {config.BATCH_SIZE}")
        logger.info(f"   Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")

        trainer.train()

        logger.info(" Treinamento concluído com sucesso!")

        # Salvamento
        logger.info(f" Salvando modelo em {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Validação
        output_path = Path(output_dir)
        if output_path.exists():
            logger.info(f" Modelo salvo com sucesso em {output_dir}")
        else:
            logger.error(f" Erro: Diretório {output_dir} não foi criado")

        return trainer

    except Exception as e:
        logger.error(f" Erro durante o treinamento: {e}")
        raise
    finally:
        # Limpeza de memória
        logger.info(" Limpando memória...")
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

# Executa o treinamento
trainer = train_model(
    model,
    dataset,
    peft_config,
    tokenizer,
    training_arguments,
    config.NEW_MODEL_NAME
)

# ===================================================================
# 8. FINALIZAÇÃO
# ===================================================================

logger.info("\n" + "="*60)
logger.info(" TREINAMENTO CONCLUÍDO COM SUCESSO!")
logger.info("="*60)
logger.info(f" Modelo salvo em: ./{config.NEW_MODEL_NAME}")
logger.info(f" Logs salvos em: training.log")
logger.info("\n Para testar o modelo, execute:")
logger.info("   python test_model.py")
logger.info("\n   Ou customize as perguntas editando test_model.py")
logger.info("="*60)
