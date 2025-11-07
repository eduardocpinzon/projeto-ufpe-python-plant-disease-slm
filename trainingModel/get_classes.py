"""
Script auxiliar para obter as classes corretas do dataset
e atualizar o predict.py se necessário
"""

import os

# Caminho para o dataset
TRAIN_DIR = "../New Plant Diseases Dataset/train"

def get_classes_from_dataset():
    """Obtém as classes diretamente do dataset"""
    if not os.path.exists(TRAIN_DIR):
        print(f"Erro: Diretório não encontrado: {TRAIN_DIR}")
        return None

    # Lista todas as pastas (classes)
    classes = sorted(os.listdir(TRAIN_DIR))

    # Remove arquivos ocultos e .DS_Store
    classes = [c for c in classes if not c.startswith('.')]

    return classes


def main():
    print("=" * 80)
    print("Obtendo classes do dataset...")
    print("=" * 80)

    classes = get_classes_from_dataset()

    if classes is None:
        return

    print(f"\nTotal de classes: {len(classes)}")
    print("\nClasses encontradas:\n")

    # Imprime em formato Python para copiar/colar
    print("CLASSES = [")
    for i, cls in enumerate(classes):
        if i < len(classes) - 1:
            print(f"    '{cls}',")
        else:
            print(f"    '{cls}'")
    print("]")

    print("\n" + "=" * 80)
    print("\nCopie a lista acima e cole no predict.py na variável CLASSES")
    print("=" * 80)


if __name__ == "__main__":
    main()

