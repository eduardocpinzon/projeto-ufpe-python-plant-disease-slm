"""
Script de Teste para Modelo Fine-tuned Phi-3
Permite testar o modelo treinado de forma independente

USO:
    python test_model.py

CONFIGURAÇÃO:
    - Ajuste MODEL_PATH para o caminho do seu modelo treinado
    - Adicione suas perguntas em TEST_QUESTIONS
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from peft import PeftModel
import logging
import gc
from pathlib import Path
from typing import List, Dict, Optional

# ===================================================================
# CONFIGURAÇÃO DE LOGGING
# ===================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ===================================================================
# CLASSE DE TESTE DO MODELO
# ===================================================================
class ModelTester:
    """
    Classe para testar modelos fine-tuned Phi-3.

    Permite carregar e testar modelos de forma independente do treinamento.

    Attributes:
        base_model_name: Nome do modelo base no HuggingFace
        adapter_path: Caminho para o adaptador LoRA
        device: Device de computação ('mps', 'cuda', 'cpu')
        model: Modelo carregado
        tokenizer: Tokenizador
    """

    def __init__(
        self,
        base_model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        adapter_path: str = "./phi3-mini-doencas-agricolas-mps",
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """
        Inicializa o testador de modelo.

        Args:
            base_model_name: Nome do modelo base
            adapter_path: Caminho para o adaptador LoRA treinado
            max_new_tokens: Máximo de tokens a gerar
            temperature: Temperatura para sampling (0.0 = determinístico, 1.0 = criativo)
            top_p: Nucleus sampling threshold
        """
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Detecta device
        self.device = self._setup_device()

        # Inicializa como None
        self.model = None
        self.tokenizer = None

        logger.info(" ModelTester inicializado")
        logger.info(f"   Modelo base: {base_model_name}")
        logger.info(f"   Adaptador: {adapter_path}")
        logger.info(f"   Device: {self.device}")

    def _setup_device(self) -> str:
        """
        Detecta e configura o dispositivo de computação.

        Returns:
            Device string ('mps', 'cuda', ou 'cpu')
        """
        if torch.backends.mps.is_available():
            try:
                test_tensor = torch.zeros(1, device='mps')
                del test_tensor
                logger.info(" MPS disponível e funcional (Apple Silicon)")
                return "mps"
            except Exception as e:
                logger.warning(f" MPS com problemas: {e}")
                return "cpu"
        elif torch.cuda.is_available():
            logger.info(" CUDA disponível (GPU NVIDIA)")
            return "cuda"
        else:
            logger.info(" Usando CPU")
            return "cpu"

    def load_model(self):
        """
        Carrega o modelo base e aplica o adaptador LoRA.

        Raises:
            FileNotFoundError: Se o adaptador não for encontrado
            Exception: Se houver erro no carregamento
        """
        logger.info(" Carregando modelo...")

        # Valida que o adaptador existe
        adapter_path = Path(self.adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adaptador não encontrado em: {self.adapter_path}\n"
                f"Execute o treinamento primeiro ou ajuste o caminho."
            )

        try:
            # Carrega modelo base
            logger.info(f"   Carregando modelo base: {self.base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )

            # Carrega tokenizador
            logger.info("   Carregando tokenizador...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Carrega adaptador LoRA
            logger.info(f"   Carregando adaptador LoRA: {self.adapter_path}")
            model_adaptado = PeftModel.from_pretrained(base_model, self.adapter_path)

            # Merge dos pesos para inferência mais rápida
            logger.info("   Fazendo merge dos pesos...")
            self.model = model_adaptado.merge_and_unload()
            self.model.to(self.device)

            logger.info(" Modelo carregado e pronto para inferência")

        except Exception as e:
            logger.error(f" Erro ao carregar modelo: {e}")
            raise

    def test_question(
            self,
            question: str,
            verbose: bool = True
    ) -> str:
        """
        Testa o modelo com uma pergunta usando geração direta.

        CORREÇÃO: Usa model.generate() ao invés de pipeline para evitar
        problemas com DynamicCache no MPS.

        Args:
            question: Pergunta a fazer ao modelo
            verbose: Se True, mostra logs detalhados

        Returns:
            Resposta gerada pelo modelo

        Raises:
            RuntimeError: Se o modelo não foi carregado
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Modelo não carregado. Execute load_model() primeiro."
            )

        if verbose:
            logger.info(f" Testando: {question}")

        try:
            # Formata prompt no estilo Phi-3
            prompt_formatado = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"

            # Tokeniza entrada
            inputs = self.tokenizer(
                prompt_formatado,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Gera resposta com model.generate() diretamente
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=False,  # Desabilita cache para evitar erro
                )

            # Decodifica resposta
            resposta_completa = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Extrai apenas a parte do assistente
            if "<|assistant|>\n" in resposta_completa:
                partes = resposta_completa.split("<|assistant|>\n")
                resposta = partes[-1].split("<|end|>")[0].strip()
            else:
                resposta = resposta_completa.strip()

            if verbose:
                logger.info(f" Resposta: {resposta}")

            return resposta

        except Exception as e:
            logger.error(f" Erro durante inferência: {e}")
            raise

    def test_multiple(
        self,
        questions: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, str]]:
        """
        Testa o modelo com múltiplas perguntas.

        Args:
            questions: Lista de perguntas
            show_progress: Se True, mostra progresso

        Returns:
            Lista de dicionários com 'question' e 'answer'
        """
        resultados = []

        if show_progress:
            logger.info("\n" + "="*60)
            logger.info(f"TESTANDO {len(questions)} PERGUNTAS")
            logger.info("="*60)

        for i, pergunta in enumerate(questions, 1):
            if show_progress:
                logger.info(f"\n TESTE {i}/{len(questions)}")
                logger.info(f"PERGUNTA: {pergunta}")

            resposta = self.test_question(pergunta, verbose=False)

            if show_progress:
                logger.info(f"RESPOSTA: {resposta}")
                logger.info("-" * 60)

            resultados.append({
                'question': pergunta,
                'answer': resposta
            })

        if show_progress:
            logger.info("\n✅ Todos os testes concluídos!")

        return resultados

    def cleanup(self):
        """
        Libera memória do modelo.
        """
        logger.info(" Limpando memória...")

        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()

        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info(" Memória liberada")


# ===================================================================
# FUNÇÃO PRINCIPAL PARA EXECUÇÃO INDEPENDENTE
# ===================================================================
def main():
    """
    Função principal para testar o modelo.

    Personalize as perguntas e configurações aqui.
    """
    # CONFIGURAÇÃO: Ajuste estes valores conforme necessário
    BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
    ADAPTER_PATH = "./phi3-mini-doencas-agricolas-mps"

    # PERGUNTAS DE TESTE: Adicione suas próprias perguntas aqui
    TEST_QUESTIONS = [
        "Quais são as recomendações para o tratamento de 'Esca (Black Measles)' em Uva?",
        "Quais as doenças comuns que afetam o cultivo de milho?",
    ]

    # Parâmetros de geração
    MAX_NEW_TOKENS = 150
    TEMPERATURE = 0.7
    TOP_P = 0.9

    try:
        # Cria testador
        tester = ModelTester(
            base_model_name=BASE_MODEL,
            adapter_path=ADAPTER_PATH,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )

        # Carrega modelo
        tester.load_model()

        # Testa múltiplas perguntas
        resultados = tester.test_multiple(TEST_QUESTIONS)

        # Salva resultados (opcional)
        logger.info("\n" + "="*60)
        logger.info("RESUMO DOS RESULTADOS")
        logger.info("="*60)
        for i, res in enumerate(resultados, 1):
            logger.info(f"\n{i}. {res['question']}")
            logger.info(f"   → {res['answer']}")

        # Limpa memória
        tester.cleanup()
        try:
            # Cria testador
            tester = ModelTester(
                base_model_name=BASE_MODEL,
                adapter_path=ADAPTER_PATH,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P
            )

            # Carrega modelo
            tester.load_model()

            # Testa múltiplas perguntas
            resultados = tester.test_multiple(TEST_QUESTIONS)

            # Salva resultados (opcional)
            logger.info("\n" + "=" * 60)
            logger.info("RESUMO DOS RESULTADOS")
            logger.info("=" * 60)
            for i, res in enumerate(resultados, 1):
                logger.info(f"\n{i}. {res['question']}")
                logger.info(f"   → {res['answer']}")

            # Limpa memória
            tester.cleanup()

            logger.info(f"\n Modelo testado: {ADAPTER_PATH}")
            logger.info(" Logs salvos em: test_model.log")

        except FileNotFoundError as e:
            logger.error(f"\n {e}")

        except Exception as e:
            logger.error(f"\n Erro durante teste: {e}")
            raise

        logger.info(f"\n Modelo testado: {ADAPTER_PATH}")
        logger.info(" Logs salvos em: test_model.log")

    except FileNotFoundError as e:
        logger.error(f"\n {e}")

    except Exception as e:
        logger.error(f"\n Erro durante teste: {e}")
        raise


# ===================================================================
# EXECUÇÃO DO SCRIPT
# ===================================================================
if __name__ == "__main__":
    main()

