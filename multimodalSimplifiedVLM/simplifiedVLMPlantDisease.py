"""
Simplified VLM
Combina CNN (classificação) + SLM (geração de texto)
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys

# Adiciona trainingModel ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importa ResNet9
from trainingModel.plant_disease_classification import ResNet9

from trainingModel.get_classes import get_classes_from_dataset

CNN_MODEL_PATH = '../trainingModel/output/plant-disease-model-complete.pth'
SLM_MODEL_PATH = '../slm/phi3-mini-doencas-agricolas-mps'

class AgriculturalVLM:
    """VLM para diagnóstico agrícola com imagem + texto"""

    def __init__(
            self,
            cnn_model_path: str = CNN_MODEL_PATH,
            slm_model_path: str = SLM_MODEL_PATH,
            device: str = 'mps',
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.9
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # 1. Carrega CNN (classificador)
        self.cnn_model = self._load_cnn(cnn_model_path)
        self.cnn_classes = self._load_classes()

        # 2. Carrega SLM (gerador de texto)
        self.slm_model, self.tokenizer = self._load_slm(slm_model_path)

        # Transform para imagens
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_cnn(self, model_path: str):
        """Carrega modelo CNN treinado"""
        model = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        model.to(self.device)
        model.eval()
        return model

    def _load_slm(self, model_path: str):
        """Carrega SLM fine-tuned"""
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            revision="main"
        )

        # Desabilitar cache para evitar erro com DynamicCache
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            revision="main"
        )

        model.eval()
        return model, tokenizer

    def _load_classes(self):
        """Carrega lista de classes do dataset"""
        # Adapte para seu caminho
        train_dir = Path('../New Plant Diseases Dataset/train')
        if train_dir.exists():
            classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        else:
            # Fallback para classes hard-coded
            classes = get_classes_from_dataset()
        return classes

    def classify_image(self, image_path: str) -> tuple:
        """
        Classifica imagem e retorna predição + confiança

        Returns:
            (predicted_class, confidence, top_5_predictions)
        """
        # Carrega e preprocessa imagem
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Inferência
        with torch.no_grad():
            output = self.cnn_model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)

            # Top-1
            confidence, pred_idx = torch.max(probs, dim=1)
            predicted_class = self.cnn_classes[pred_idx.item()]

            # Top-5
            top5_prob, top5_idx = torch.topk(probs, 5)
            top5_predictions = [
                (self.cnn_classes[idx], prob.item())
                for idx, prob in zip(top5_idx[0], top5_prob[0])
            ]

        return predicted_class, confidence.item(), top5_predictions

    def generate_diagnosis(
            self,
            predicted_class: str,
            confidence: float,
            top5_predictions: list
    ) -> str:
        """
        Gera texto de diagnóstico usando SLM

        Args:
            predicted_class: Classe predita pela CNN
            confidence: Confiança da predição
            top5_predictions: Top-5 predições [(class, prob), ...]

        Returns:
            Texto de diagnóstico gerado
        """
        # Parse classe (ex: "Tomato___Early_blight" -> "Tomate", "Pinta Preta")
        if '___' in predicted_class:
            plant, disease = predicted_class.split('___')
            plant = plant.replace('_', ' ')
            disease = disease.replace('_', ' ')
        else:
            plant, disease = predicted_class, "Desconhecida"

        # Monta prompt para o SLM
        prompt = self._create_diagnosis_prompt(
            plant, disease, confidence, top5_predictions
        )

        # Gera resposta
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.slm_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                use_cache=False,  # Desabilita cache para evitar erros
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extrai apenas a resposta (remove prompt)
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()

        return response

    def _create_diagnosis_prompt(
            self,
            plant: str,
            disease: str,
            confidence: float,
            top5_predictions: list
    ) -> str:
        """Cria prompt estruturado para o SLM"""

        # Lista top-5 para contexto
        top5_text = "\n".join([
            f"- {cls.split('___')[1] if '___' in cls else cls}: {prob * 100:.1f}%"
            for cls, prob in top5_predictions
        ])

        prompt = f"""<|user|>
Analisei uma imagem de folha e obtive os seguintes resultados:

Planta: {plant}
Doença detectada: {disease}
Confiança: {confidence * 100:.1f}%

Top 5 diagnósticos possíveis:
{top5_text}

Com base nesta análise, forneça:
1. Confirmação ou questionamento do diagnóstico
2. Sintomas típicos desta doença
3. Recomendações de tratamento
4. Medidas preventivas

Seja técnico mas compreensível para agricultores.<|end|>
<|assistant|>
"""
        return prompt

    def diagnose_image(self, image_path: str) -> dict:
        """
        Pipeline completo: imagem -> classificação -> diagnóstico

        Args:
            image_path: Caminho da imagem

        Returns:
            Dicionário com todos os resultados
        """
        print(f"🔍 Analisando imagem: {image_path}")

        # Passo 1: Classificação CNN
        print("📸 Etapa 1: Classificação visual...")
        predicted_class, confidence, top5 = self.classify_image(image_path)

        # Passo 2: Geração de diagnóstico
        print("🤖 Etapa 2: Gerando diagnóstico com IA...")
        diagnosis_text = self.generate_diagnosis(predicted_class, confidence, top5)

        # Estrutura resultado
        result = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'diagnosis': diagnosis_text
        }

        return result

    def print_diagnosis(self, result: dict):
        """Imprime resultado formatado"""
        print("\n" + "=" * 80)
        print("🌱 DIAGNÓSTICO AGRÍCOLA COMPLETO")
        print("=" * 80)

        # Informações da imagem
        print(f"\nImagem: {result['image_path']}")

        # Classificação principal
        plant, disease = self._parse_class(result['predicted_class'])
        print(f"\nPlanta: {plant}")
        print(f"Doença: {disease}")
        print(f"Confiança: {result['confidence'] * 100:.2f}%")

        # Diagnóstico detalhado
        print("\nAnálise Detalhada:")
        print("-" * 80)
        print(result['diagnosis'])
        print("=" * 80)

    def _parse_class(self, class_name: str) -> tuple:
        """Converte 'Tomato___Early_blight' -> ('Tomate', 'Pinta Preta')"""
        if '___' in class_name:
            plant, disease = class_name.split('___')
            return plant.replace('_', ' '), disease.replace('_', ' ')
        return class_name, "Desconhecida"


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

def main():
    """Exemplo de uso do VLM"""

    # Inicializa VLM
    vlm = AgriculturalVLM(
        cnn_model_path=CNN_MODEL_PATH,
        slm_model_path=SLM_MODEL_PATH,
        device='mps'  # ou 'cuda', 'cpu'
    )

    # Analisa imagem
    image_path = '../trainingModel/test_apple.jpeg'
    result = vlm.diagnose_image(image_path)

    # Mostra resultado
    vlm.print_diagnosis(result)


if __name__ == "__main__":
    main()