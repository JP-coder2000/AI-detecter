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
    print(f"Cargando modelo desde {model_dir}...")
    
    # Cargar feature extractor
    feature_extractor_path = os.path.join(model_dir, 'feature_extractor.keras')
    if os.path.exists(feature_extractor_path):
        print(f"Cargando extractor de características desde {feature_extractor_path}")
        feature_extractor = joblib.load(feature_extractor_path)
    else:
        # Si no existe, crear uno nuevo
        print("Creando nuevo extractor de características")
        feature_extractor = FeatureExtractor()
        
        # Cargar vocabulario si existe
        vocab_path = os.path.join(model_dir, 'vocabulary.pkl')
        if os.path.exists(vocab_path):
            print(f"Cargando vocabulario desde {vocab_path}")
            feature_extractor.load_vocabulary(vocab_path)
    
    # Cargar procesador de texto
    text_processor_path = os.path.join(model_dir, 'text_processor.keras')
    if os.path.exists(text_processor_path):
        print(f"Cargando procesador de texto desde {text_processor_path}")
        text_processor = joblib.load(text_processor_path)
    else:
        # Si no existe, crear uno nuevo
        print("Creando nuevo procesador de texto")
        text_processor = TextProcessor()
    
    # Cargar SVM
    svm_path = os.path.join(model_dir, 'svm_model.keras')
    if os.path.exists(svm_path):
        print(f"Cargando modelo SVM desde {svm_path}")
        svm_model = joblib.load(svm_path)
    else:
        print("ERROR: No se encontró el modelo SVM")
        return None
    
    # Cargar CNN
    cnn_path = os.path.join(model_dir, 'cnn_model.keras')
    if os.path.exists(cnn_path):
        print(f"Cargando modelo CNN desde {cnn_path}")
        try:
            cnn_model = tf.keras.models.load_model(cnn_path)
            # Envolver en nuestro contenedor si es necesario
            if not isinstance(cnn_model, PlagiarismCNN):
                cnn_wrapper = PlagiarismCNN(vocab_size=feature_extractor.vocab_size)
                cnn_wrapper.model = cnn_model
                cnn_model = cnn_wrapper
        except Exception as e:
            print(f"Error cargando CNN: {e}")
            return None
    else:
        print("ERROR: No se encontró el modelo CNN")
        return None
    
    # Cargar configuración
    config_path = os.path.join(model_dir, 'config.keras')
    if os.path.exists(config_path):
        print(f"Cargando configuración desde {config_path}")
        config = joblib.load(config_path)
        ensemble_weights = config.get('ensemble_weights', (0.5, 0.5))
    else:
        print("Usando pesos por defecto (0.5, 0.5)")
        ensemble_weights = (0.5, 0.5)
    
    # Crear modelo híbrido
    hybrid_model = HybridPlagiarismDetector(
        svm_model=svm_model,
        cnn_model=cnn_model,
        feature_extractor=feature_extractor,
        text_processor=text_processor,
        ensemble_weights=ensemble_weights
    )
    
    print("Modelo cargado correctamente")
    return hybrid_model

def test_model(model, source_text, suspicious_text):
    """Prueba el modelo con un par de textos."""
    try:
        # Preprocesar textos
        print("\nPreprocesando textos...")
        processor = model.text_processor
        source_processed = processor.preprocess_text(source_text)['processed']
        suspicious_processed = processor.preprocess_text(suspicious_text)['processed']
        
        print("Extrayendo características...")
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
        
        print("Realizando predicción con CNN...")
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
        print(f"Error en la predicción: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Cargando modelo...")
    model = load_model()
    
    if not model:
        print("Error cargando el modelo. Abortando.")
        exit(1)
    
    # Texto original
    original = "La inteligencia artificial es un campo de la informática que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana."

    # Copia directa (debería detectarse fácilmente)
    plagio_directo = "La inteligencia artificial es un campo de la informática que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana."

    # Paráfrasis leve (debería detectarse bien)
    plagio_leve = "La IA es una rama de las ciencias computacionales que intenta desarrollar sistemas que pueden ejecutar tareas que típicamente necesitan inteligencia humana."

    # Paráfrasis fuerte (prueba más difícil)
    plagio_fuerte = "Los sistemas de aprendizaje automatizado son tecnologías informáticas diseñadas para emular capacidades cognitivas humanas en la resolución de problemas complejos."

    # Texto no relacionado (no debería detectarse como plagio)
    no_plagio = "Los avances en tecnología médica han permitido desarrollar nuevos tratamientos para enfermedades previamente incurables, mejorando la calidad de vida de millones de personas."

    # Probar cada caso
    test_cases = [
        ("Copia directa", original, plagio_directo),
        ("Paráfrasis leve", original, plagio_leve),
        ("Paráfrasis fuerte", original, plagio_fuerte),
        ("Texto no relacionado", original, no_plagio)
    ]
    
    for description, source, suspicious in test_cases:
        print(f"\n{'='*50}")
        print(f"CASO DE PRUEBA: {description}")
        print(f"Texto original: {source[:100]}...")
        print(f"Texto sospechoso: {suspicious[:100]}...")
        
        resultado = test_model(model, source, suspicious)
        
        if resultado:
            print(f"\nRESULTADO:")
            print(f"¿Es plagio?: {resultado['is_plagiarism']}")
            print(f"Confianza: {resultado['confidence']:.4f}")
            print(f"Score SVM: {resultado['svm_score']:.4f}")
            print(f"Score CNN: {resultado['cnn_score']:.4f}")
            print(f"Tipo de clon: {resultado['clone_type']}")
            print(f"Similitud Jaccard: {resultado['jaccard_similarity']:.4f}")
        else:
            print("Error al procesar este caso.")