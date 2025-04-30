import os
import numpy as np
import joblib
import tensorflow as tf
import re
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

def preprocess_code(code):
    """
    Preprocesa código fuente para comparación de plagio.
    
    Realiza:
    1. Eliminación de comentarios
    2. Normalización de nombres de variables/funciones 
    3. Eliminación de espacios en blanco excesivos
    """
    # Eliminar comentarios de una línea (estilo Python)
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    
    # Eliminar docstrings y comentarios multilínea
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    
    # Palabras reservadas de Python (que no queremos normalizar)
    keywords = ['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 
                'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 
                'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 
                'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 
                'try', 'while', 'with', 'yield']
    
    # Encontrar identificadores definidos por el usuario
    identifiers = {}
    
    # Normalizar nombres de funciones 
    func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    for i, func_name in enumerate(set(re.findall(func_pattern, code))):
        if func_name not in keywords:
            identifiers[func_name] = f"FUNC_{i}"
    
    # Normalizar nombres de variables
    var_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*='
    for i, var_name in enumerate(set(re.findall(var_pattern, code))):
        if var_name not in keywords and var_name not in identifiers:
            identifiers[var_name] = f"VAR_{i}"
    
    # Aplicar sustituciones (cuidando no reemplazar subcadenas)
    for name, placeholder in identifiers.items():
        code = re.sub(r'\b' + name + r'\b', placeholder, code)
    
    # Normalizar espacios en blanco
    code = re.sub(r'\s+', ' ', code)
    
    return code.strip()

def test_code_plagiarism(model, source_code, suspicious_code):
    """Prueba el modelo con un par de ejemplos de código."""
    try:
        # Preprocesar el código
        print("\nPreprocesando código...")
        source_processed = preprocess_code(source_code)
        suspicious_processed = preprocess_code(suspicious_code)
        
        print("Código original preprocesado:", source_processed[:100] + "..." if len(source_processed) > 100 else source_processed)
        print("Código sospechoso preprocesado:", suspicious_processed[:100] + "..." if len(suspicious_processed) > 100 else suspicious_processed)
        
        # Usar el modelo existente con el código preprocesado
        resultado = test_model(model, source_processed, suspicious_processed)
        return resultado
        
    except Exception as e:
        print(f"Error procesando el código: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model(model, source_text, suspicious_text):
    """Prueba el modelo con un par de textos."""
    try:
        # No necesitamos preprocesar otra vez aquí
        source_processed = source_text
        suspicious_processed = suspicious_text
        
        print("Extrayendo características...")
        # Extraer características
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
        
        print("Realizando predicción con modelo híbrido...")
        # Predicción CNN
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
    
    # Ejemplo 1: Código original
    original_code = """
def factorial(n):
    Calcula el factorial de n
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)
    
# Probar la función
result = factorial(5)
print(f"El factorial de 5 es {result}")
"""

    # Ejemplo 2: Plagio directo con comentarios modificados
    direct_plagiarism = """
def factorial(n):
    # Esta función calcula n!
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)
    
# Prueba
result = factorial(5)
print(f"5! es igual a {result}")
"""

    # Ejemplo 3: Cambio de nombres de variables/funciones
    variable_change = """
def calcular_factorial(numero):
    if numero == 0 or numero == 1:
        return 1
    else:
        return numero * calcular_factorial(numero - 1)
    
# Prueba
respuesta = calcular_factorial(5)
print(f"El factorial de 5 es {respuesta}")
"""

    # Ejemplo 4: Mismo algoritmo, implementación diferente
    algorithm_change = """
def factorial(num):
    resultado = 1
    for i in range(1, num + 1):
        resultado *= i
    return resultado

# Prueba
print(f"El factorial de 5 es {factorial(5)}")
"""

    # Ejemplo 5: Código no relacionado
    unrelated_code = """
def es_primo(numero):
    if numero <= 1:
        return False
    for i in range(2, int(numero**0.5) + 1):
        if numero % i == 0:
            return False
    return True

# Verificar si 17 es primo
print(f"¿Es 17 primo? {es_primo(17)}")
"""

    # Probar cada caso
    test_cases = [
        ("Plagio Directo", original_code, direct_plagiarism),
        ("Cambio de Variables", original_code, variable_change),
        ("Cambio de Implementación", original_code, algorithm_change),
        ("Código No Relacionado", original_code, unrelated_code)
    ]
    
    for description, source, suspicious in test_cases:
        print(f"\n{'='*50}")
        print(f"CASO DE PRUEBA: {description}")
        print(f"Código original: {source[:100]}...")
        print(f"Código sospechoso: {suspicious[:100]}...")
        
        resultado = test_code_plagiarism(model, source, suspicious)
        
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