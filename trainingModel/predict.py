"""
Plant Disease Prediction Script
================================

This script loads a trained model and makes predictions on new images.

Usage:
    python predict.py path/to/image.jpg
    
Or edit the IMAGE_PATH variable below and run:
    python predict.py
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# Import from the training script
from plant_disease_classification import ResNet9, get_default_device, to_device


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to your trained model
MODEL_PATH = './plant-disease-model-complete.pth'

# Path to image for prediction (can be overridden by command line argument)
IMAGE_PATH = './test_apple.jpeg'

# Path to dataset (to get classes dynamically)
TRAIN_DIR = '../New Plant Diseases Dataset/train'

# Expected classes (fallback if dataset is not available)
# These should match the classes your model was trained on
CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]


def get_classes():
    """Get classes from dataset directory or use fallback"""
    if os.path.exists(TRAIN_DIR):
        classes = sorted(os.listdir(TRAIN_DIR))
        # Remove hidden files and .DS_Store
        classes = [c for c in classes if not c.startswith('.')]
        print(f"Loaded {len(classes)} classes from dataset directory")
        return classes
    else:
        print(f"Dataset directory not found, using predefined classes ({len(CLASSES)} classes)")
        return CLASSES


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def load_model(model_path, device):
    """Load the trained model"""
    try:
        # Try loading the complete model with weights_only=False (PyTorch 2.6+)
        model = torch.load(model_path, map_location=device, weights_only=False)
        print(f"Loaded complete model from {model_path}")
    except Exception as e:
        # If that fails, load state dict
        print(f"Could not load complete model, trying state dict... ({e})")
        model = ResNet9(3, len(CLASSES))
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        print(f"Loaded model state dict from {model_path}")
    
    model = to_device(model, device)
    model.eval()
    return model


def load_image(image_path):
    """Load and preprocess an image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Transform to tensor with ImageNet normalization (same as training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image)
    return image_tensor


def predict_image(model, image_tensor, classes, device):
    """Make prediction on a single image"""
    # Add batch dimension
    image_batch = image_tensor.unsqueeze(0)
    image_batch = to_device(image_batch, device)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_batch)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
    
    predicted_class = classes[predicted_idx.item()]
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score


def get_top_k_predictions(model, image_tensor, classes, device, k=5):
    """Get top K predictions with confidence scores"""
    # Add batch dimension
    image_batch = image_tensor.unsqueeze(0)
    image_batch = to_device(image_batch, device)
    
    # Get predictions
    with torch.no_grad():
        output = model(image_batch)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_prob, top_indices = torch.topk(probabilities, k)
    
    results = []
    for i in range(k):
        class_name = classes[top_indices[0][i].item()]
        confidence = top_prob[0][i].item()
        results.append((class_name, confidence))
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main prediction pipeline"""
    
    print("=" * 80)
    print("Plant Disease Prediction")
    print("=" * 80)
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = IMAGE_PATH
    
    print(f"\nImage path: {image_path}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model not found at {MODEL_PATH}")
        print("Please train the model first using plant_disease_classification.py")
        return
    
    # Get device
    device = get_default_device()
    print(f"Using device: {device}")
    
    # Get classes
    classes = get_classes()

    # Load model
    print("\nLoading model...")
    model = load_model(MODEL_PATH, device)
    
    # Load image
    print("Loading image...")
    try:
        image_tensor = load_image(image_path)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return
    except Exception as e:
        print(f"\nError loading image: {e}")
        return
    
    # Make prediction
    print("Making prediction...")
    predicted_class, confidence = predict_image(model, image_tensor, classes, device)

    # Display results
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print(f"\nPredicted Class: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")
    
    # Get top 5 predictions
    print("\nTop 5 Predictions:")
    print("-" * 80)
    top_predictions = get_top_k_predictions(model, image_tensor, classes, device, k=5)

    for i, (class_name, conf) in enumerate(top_predictions, 1):
        # Parse class name for better display
        if '___' in class_name:
            plant, disease = class_name.split('___')
            plant = plant.replace('_', ' ')
            disease = disease.replace('_', ' ')

            print(f"{i}. Plant: {plant}")
            print(f"   Disease: {disease}")
            print(f"   Confidence: {conf * 100:.2f}%")
            print()
        else:
            print(f"{i}. {class_name}")
            print(f"   Confidence: {conf * 100:.2f}%")
            print()

    print("=" * 80)


if __name__ == "__main__":
    main()
