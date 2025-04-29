import os
import numpy as np
import joblib
import tensorflow as tf
from preprocessing.text_processor import TextProcessor
from features.feature_extractor import FeatureExtractor
from models.svm_model import PlagiarismSVM
from models.cnn_model import PlagiarismCNN
from integration.hybrid_model import HybridPlagiarismDetector

def load_model(model_dir='models/hybrid_model'):
    """Carga el modelo híbrido desde el directorio especificado."""
    print(f"Loading model from {model_dir}...")
    
    # Cargar feature extractor
    feature_extractor_path = os.path.join(model_dir, 'feature_extractor.keras')
    if os.path.exists(feature_extractor_path):
        print(f"Loading feature extractor from {feature_extractor_path}")
        feature_extractor = joblib.load(feature_extractor_path)
    else:
        # Si no existe, crear uno nuevo
        print("Creating new feature extractor")
        feature_extractor = FeatureExtractor()
        
        # Cargar vocabulario si existe
        vocab_path = os.path.join(model_dir, 'vocabulary.pkl')
        if os.path.exists(vocab_path):
            print(f"Loading vocabulary from {vocab_path}")
            feature_extractor.load_vocabulary(vocab_path)
    
    # Cargar procesador de texto
    text_processor_path = os.path.join(model_dir, 'text_processor.keras')
    if os.path.exists(text_processor_path):
        print(f"Loading text processor from {text_processor_path}")
        text_processor = joblib.load(text_processor_path)
    else:
        # Si no existe, crear uno nuevo
        print("Creating new text processor")
        text_processor = TextProcessor()
    
    # Cargar SVM
    svm_path = os.path.join(model_dir, 'svm_model.keras')
    if os.path.exists(svm_path):
        print(f"Loading SVM model from {svm_path}")
        svm_model = joblib.load(svm_path)
    else:
        print("ERROR: SVM model not found")
        return None
    
    # Cargar CNN
    cnn_path = os.path.join(model_dir, 'cnn_model.keras')
    if os.path.exists(cnn_path):
        print(f"Loading CNN model from {cnn_path}")
        try:
            cnn_model = tf.keras.models.load_model(cnn_path)
            # Envolver en nuestro contenedor si es necesario
            if not isinstance(cnn_model, PlagiarismCNN):
                cnn_wrapper = PlagiarismCNN(vocab_size=feature_extractor.vocab_size)
                cnn_wrapper.model = cnn_model
                cnn_model = cnn_wrapper
        except Exception as e:
            print(f"Error loading CNN: {e}")
            return None
    else:
        print("ERROR: CNN model not found")
        return None
    
    # Cargar configuración
    config_path = os.path.join(model_dir, 'config.keras')
    if os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        config = joblib.load(config_path)
        ensemble_weights = config.get('ensemble_weights', (0.5, 0.5))
    else:
        print("Using default weights (0.5, 0.5)")
        ensemble_weights = (0.5, 0.5)
    
    # Crear modelo híbrido
    hybrid_model = HybridPlagiarismDetector(
        svm_model=svm_model,
        cnn_model=cnn_model,
        feature_extractor=feature_extractor,
        text_processor=text_processor,
        ensemble_weights=ensemble_weights
    )
    
    print("Model loaded successfully")
    return hybrid_model

def test_model(model, source_text, suspicious_text):
    """Prueba el modelo con un par de textos."""
    try:
        # Preprocesar textos
        print("\nPreprocessing texts...")
        processor = model.text_processor
        source_processed = processor.preprocess_text(source_text)['processed']
        suspicious_processed = processor.preprocess_text(suspicious_text)['processed']
        
        print("Extracting features...")
        # Extraer características manualmente
        features = model.feature_extractor.extract_features_for_fragment_pair(
            source_processed, 
            suspicious_processed, 
            update_vocab=False
        )
        
        # Predicción SVM
        svm_features = features['svm_features'].reshape(1, -1)
        svm_prediction = model.svm_model.predict_proba(svm_features)[0, 1]
        
        # Predicción CNN
        cnn_input = features['cnn_input']
        source_sequence = cnn_input['source_sequence'].reshape(1, -1)
        suspicious_sequence = cnn_input['suspicious_sequence'].reshape(1, -1)
        
        print("Making prediction with CNN...")
        # Llamar al modelo directamente
        cnn_prediction = model.cnn_model.model.predict(
            [source_sequence, suspicious_sequence], 
            verbose=0
        )[0][0]
        
        # Combinación ponderada
        w_svm, w_cnn = model.ensemble_weights
        ensemble_score = w_svm * svm_prediction + w_cnn * cnn_prediction
        
        # Determinar clase y nivel de confianza
        is_plagiarism = ensemble_score >= 0.5
        
        # Inferir tipo de clon
        jaccard_sim = model.feature_extractor._jaccard_similarity(source_processed, suspicious_processed)
        
        if jaccard_sim > 0.8 and ensemble_score > 0.9:
            clone_type = "Type I"  # Clones casi idénticos
        elif jaccard_sim > 0.5 and ensemble_score > 0.7:
            clone_type = "Type II"  # Cambios en identificadores/valores
        elif ensemble_score > 0.5:
            clone_type = "Type III"  # Paráfrasis más complejas
        else:
            clone_type = "Not a clone"
            
        return {
            'is_plagiarism': bool(is_plagiarism),
            'confidence': float(ensemble_score),
            'svm_score': float(svm_prediction),
            'cnn_score': float(cnn_prediction),
            'clone_type': clone_type,
            'jaccard_similarity': jaccard_sim
        }
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Loading model...")
    model = load_model()
    
    if not model:
        print("Error loading model. Aborting.")
        exit(1)
    
    # Original text
    original = "Artificial intelligence is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence."

    # Direct copy (should be easily detected)
    direct_plagiarism = "Artificial intelligence is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence."

    # Light paraphrase (should be well detected)
    light_paraphrase = "AI is a branch of computing that aims to develop systems able to execute tasks that normally need human intelligence."

    # Heavy paraphrase (harder test)
    heavy_paraphrase = "Machine learning systems are computational technologies designed to emulate human cognitive abilities in solving complex problems."

    # Unrelated text (should not be detected as plagiarism)
    non_plagiarism = "Recent advances in medical technology have enabled the development of new treatments for previously incurable diseases, improving the quality of life for millions of people."

    # Academic text (complex technical language)
    academic = "The implementation of neural networks in natural language processing has significantly enhanced the capacity for semantic understanding, facilitating more nuanced interpretations of syntactic structures."

    # Test each case
    test_cases = [
        ("Direct copy", original, direct_plagiarism),
        ("Light paraphrase", original, light_paraphrase),
        ("Heavy paraphrase", original, heavy_paraphrase),
        ("Unrelated text", original, non_plagiarism),
        ("Related academic", original, academic)
    ]
    
    for description, source, suspicious in test_cases:
        print(f"\n{'='*50}")
        print(f"TEST CASE: {description}")
        print(f"Original text: {source[:100]}...")
        print(f"Suspicious text: {suspicious[:100]}...")
        
        resultado = test_model(model, source, suspicious)
        
        if resultado:
            print(f"\nRESULT:")
            print(f"Is plagiarism?: {resultado['is_plagiarism']}")
            print(f"Confidence: {resultado['confidence']:.4f}")
            print(f"SVM score: {resultado['svm_score']:.4f}")
            print(f"CNN score: {resultado['cnn_score']:.4f}")
            print(f"Clone type: {resultado['clone_type']}")
            print(f"Jaccard similarity: {resultado['jaccard_similarity']:.4f}")
        else:
            print("Error processing this case.")