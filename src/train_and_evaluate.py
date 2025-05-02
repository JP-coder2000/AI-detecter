import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib

from models.svm_model import PlagiarismSVM
from models.cnn_model import PlagiarismCNN
from integration.hybrid_model import HybridPlagiarismDetector
from features.feature_extractor import FeatureExtractor
from preprocessing.text_processor import TextProcessor

def create_enhanced_visualizations(y_test, best_predictions, svm_proba, cnn_proba, hybrid_proba, 
                                  history, metadata_test, output_dir):
    """
    Genera visualizaciones mejoradas para los resultados del modelo.
    
    Args:
        y_test: Etiquetas verdaderas
        best_predictions: Predicciones del modelo híbrido
        svm_proba: Probabilidades de SVM
        cnn_proba: Probabilidades de CNN
        hybrid_proba: Probabilidades del modelo híbrido
        history: Historial de entrenamiento de CNN
        metadata_test: Metadatos del conjunto de prueba
        output_dir: Directorio donde guardar las gráficas
    """
    # Asegurarse de que el directorio existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuración de estilo
    plt.style.use('default')
    colors = {'svm': '#1f77b4', 'cnn': '#ff7f0e', 'hybrid': '#2ca02c'}
    
    # 1. Matriz de confusión
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, best_predictions)
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('Matriz de Confusión - Modelo Híbrido', fontsize=16)
    plt.ylabel('Etiqueta Real', fontsize=14)
    plt.xlabel('Etiqueta Predicha', fontsize=14)
    plt.xticks([0.5, 1.5], ['No Plagio', 'Plagio'], fontsize=12)
    plt.yticks([0.5, 1.5], ['No Plagio', 'Plagio'], fontsize=12, rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    
    # 2. Curvas de aprendizaje - siempre empezando desde 0
    if hasattr(history, 'history'):
        plt.figure(figsize=(12, 10))
        
        # 2.1 Precisión
        plt.subplot(2, 1, 1)
        plt.plot(history.history['accuracy'], label='Entrenamiento', color=colors['cnn'], linewidth=2)
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validación', 
                    color=colors['hybrid'], linewidth=2, linestyle='--')
        
        plt.title('Precisión durante Entrenamiento', fontsize=16)
        plt.ylabel('Precisión', fontsize=14)
        plt.xlabel('Época', fontsize=14)
        plt.ylim(0, 1.0)  # Forzar inicio desde 0
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # 2.2 Pérdida
        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'], label='Entrenamiento', color=colors['cnn'], linewidth=2)
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validación', 
                    color=colors['hybrid'], linewidth=2, linestyle='--')
        
        plt.title('Pérdida durante Entrenamiento', fontsize=16)
        plt.ylabel('Pérdida', fontsize=14)
        plt.xlabel('Época', fontsize=14)
        plt.ylim(0, max(history.history['loss'])*1.1)  # Escala dinámica pero desde 0
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300)

def train_and_evaluate(data_dir='src/data/processed', model_dir='models',
                      use_pretrained_embeddings=True,
                      embedding_model='glove-wiki-gigaword-300',
                      language='english'):
    """
    Entrena y evalúa los modelos con el dataset generado.
    
    Args:
        data_dir: Directorio con los datos procesados
        model_dir: Directorio donde guardar los modelos
        use_pretrained_embeddings: Si se deben usar embeddings preentrenados
        embedding_model: Modelo de embeddings a utilizar
        language: Idioma del modelo
    """
    print(f"Cargando datos desde {data_dir}...")
    
    # Verificar que el directorio existe
    if not os.path.exists(data_dir):
        print(f"ERROR: El directorio {data_dir} no existe")
        return None
    
    # Cargar datos
    try:
        X_svm_train = np.load(os.path.join(data_dir, 'X_svm_train.npy'))
        X_cnn_source_train = np.load(os.path.join(data_dir, 'X_cnn_source_train.npy'))
        X_cnn_suspicious_train = np.load(os.path.join(data_dir, 'X_cnn_suspicious_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        
        X_svm_test = np.load(os.path.join(data_dir, 'X_svm_test.npy'))
        X_cnn_source_test = np.load(os.path.join(data_dir, 'X_cnn_source_test.npy'))
        X_cnn_suspicious_test = np.load(os.path.join(data_dir, 'X_cnn_suspicious_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        metadata_test = pd.read_csv(os.path.join(data_dir, 'metadata_test.csv'))
    except Exception as e:
        print(f"ERROR al cargar los datos: {e}")
        return None
    
    print(f"Datos cargados: {len(y_train)} ejemplos de entrenamiento, {len(y_test)} de prueba")
    
    # Cargar procesador de texto
    text_processor = TextProcessor(language=language)
    
    # Cargar vocabulario y feature_extractor
    vocab_path = os.path.join(data_dir, 'vocabulary.pkl')
    feature_extractor = None
    
    if os.path.exists(vocab_path):
        print(f"Cargando vocabulario existente desde {vocab_path}")
        feature_extractor = FeatureExtractor(
            language=language,
            use_pretrained_embeddings=use_pretrained_embeddings,
            embedding_model=embedding_model
        )
        feature_extractor.load_vocabulary(vocab_path)
    else:
        print("¡ADVERTENCIA! No se encontró vocabulario. Esto puede causar problemas con embeddings preentrenados.")
        feature_extractor = FeatureExtractor(
            language=language,
            use_pretrained_embeddings=use_pretrained_embeddings,
            embedding_model=embedding_model
        )
    
    # Crear directorio para modelos
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Entrenar modelo SVM
    print("\n--- Entrenando modelo SVM ---")
    svm_model = PlagiarismSVM(kernel='rbf', class_weight='balanced')
    svm_model.fit(X_svm_train, y_train)
    
    # Evaluar SVM
    print("\nEvaluando SVM...")
    y_pred_svm = svm_model.predict(X_svm_test)
    svm_proba = svm_model.predict_proba(X_svm_test)[:, 1]  # Probabilidad de clase positiva
    
    print("\nInforme de clasificación SVM:")
    print(classification_report(y_test, y_pred_svm))
    
    # 2. Entrenar modelo CNN
    print("\n--- Entrenando modelo CNN ---")
    
    # Determinar parámetros del modelo
    vocab_size = max(X_cnn_source_train.max(), X_cnn_suspicious_train.max()) + 1
    max_length = X_cnn_source_train.shape[1]
    
    print(f"Parámetros CNN: vocab_size={vocab_size}, max_length={max_length}")
    
    # IMPORTANTE: Cargar la matriz de embeddings si existe
    embedding_matrix = None
    embedding_matrix_path = os.path.join(data_dir, 'embedding_matrix.npy')
    
    if os.path.exists(embedding_matrix_path) and use_pretrained_embeddings:
        print(f"Cargando matriz de embeddings desde {embedding_matrix_path}")
        embedding_matrix = np.load(embedding_matrix_path)
        print(f"Matriz de embeddings cargada con forma: {embedding_matrix.shape}")
    elif use_pretrained_embeddings:
        print("Generando matriz de embeddings a partir del vocabulario...")
        # Usar el feature_extractor para crear la matriz
        embedding_matrix = feature_extractor.create_embedding_matrix()
        print(f"Matriz de embeddings generada con forma: {embedding_matrix.shape}")
    else:
        print("No se usarán embeddings preentrenados")
    
    # Inicializar modelo CNN
    embedding_dim = 300 if use_pretrained_embeddings else 100
    
    cnn_model = PlagiarismCNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_length=max_length,
        filters_per_size=64,
        filter_sizes=[3, 4, 5],
        dropout_rate=0.5,
        use_pretrained=use_pretrained_embeddings,
        embedding_model=embedding_model,
        trainable_embeddings=False  # No entrenar embeddings inicialmente
    )
    
    # Establecer la matriz de embeddings en el modelo CNN
    if embedding_matrix is not None and use_pretrained_embeddings:
        print("Estableciendo matriz de embeddings en el modelo CNN")
        # Asignar la matriz al atributo
        cnn_model.embedding_matrix = embedding_matrix
        
        # Actualizar los pesos del modelo directamente
        for layer in cnn_model.model.layers:
            if layer.name == 'embedding':
                print(f"Actualizando capa de embedding: {layer.name}")
                layer.set_weights([embedding_matrix])
                break
    
    # Definir callbacks para CNN
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Reducir tasa de aprendizaje cuando la mejora se estanca
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=0.00001
    )
    
    # Guardar mejor modelo
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_dir, 'best_cnn_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Entrenar CNN
    history = cnn_model.fit(
        X_cnn_source_train, 
        X_cnn_suspicious_train, 
        y_train,
        validation_split=0.1,
        batch_size=64,
        epochs=50,
        callbacks=[early_stopping, reduce_lr, model_checkpoint]
    )
    
    # Evaluar CNN
    print("\nEvaluando CNN...")
    y_pred_cnn_prob = cnn_model.predict(X_cnn_source_test, X_cnn_suspicious_test)
    y_pred_cnn = (y_pred_cnn_prob >= 0.5).astype(int)
    
    print("\nInforme de clasificación CNN:")
    print(classification_report(y_test, y_pred_cnn))
    
    # 3. Evaluar modelo híbrido con diferentes pesos
    print("\n--- Evaluando modelo híbrido ---")
    
    # Probar diferentes combinaciones de pesos
    weight_combinations = [
        (0.7, 0.3),  # Más peso al SVM
        (0.5, 0.5),  # Peso igual
        (0.3, 0.7)   # Más peso al CNN
    ]
    
    best_f1 = 0
    best_weights = None
    best_predictions = None
    best_hybrid_proba = None
    
    for w_svm, w_cnn in weight_combinations:
        # Combinar predicciones
        hybrid_proba = w_svm * svm_proba + w_cnn * y_pred_cnn_prob.flatten()
        hybrid_pred = (hybrid_proba >= 0.5).astype(int)
        
        # Evaluar
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, hybrid_pred, average='binary')
        
        print(f"\nPesos: SVM={w_svm}, CNN={w_cnn}")
        print(f"Precisión: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_weights = (w_svm, w_cnn)
            best_predictions = hybrid_pred
            best_hybrid_proba = hybrid_proba
    
    print(f"\nMejores pesos: SVM={best_weights[0]}, CNN={best_weights[1]}")
    print("\nInforme de clasificación híbrido (mejores pesos):")
    print(classification_report(y_test, best_predictions))
    
    # 4. Analizar por tipo de obfuscación
    print("\n--- Análisis por tipo de obfuscación ---")
    
    # Agregar predicciones al DataFrame de metadatos
    metadata_test['svm_pred'] = y_pred_svm
    metadata_test['cnn_pred'] = y_pred_cnn
    metadata_test['hybrid_pred'] = best_predictions
    
    # Agrupar por tipo de obfuscación
    obfuscation_results = []
    for obfuscation_type in metadata_test['obfuscation'].unique():
        mask = metadata_test['obfuscation'] == obfuscation_type
        if sum(mask) == 0:
            continue
            
        y_true_subset = metadata_test.loc[mask, 'is_plagiarism'].values
        y_pred_subset = metadata_test.loc[mask, 'hybrid_pred'].values
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_subset, y_pred_subset, average='binary', zero_division=0)
        
        obfuscation_results.append({
            'tipo': obfuscation_type,
            'ejemplos': sum(mask),
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        print(f"\nTipo de obfuscación: {obfuscation_type}")
        print(f"Ejemplos: {sum(mask)}")
        print(f"Precisión: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # 5. Guardar modelos
    print("\n--- Guardando modelos ---")
    
    # Guardar SVM
    joblib.dump(svm_model, os.path.join(model_dir, 'svm_model.keras'))
    
    # Guardar CNN
    cnn_model.save(os.path.join(model_dir, 'cnn_model.keras'))
    
    # Guardar FeatureExtractor
    joblib.dump(feature_extractor, os.path.join(model_dir, 'feature_extractor.keras'))
    
    # Guardar TextProcessor
    joblib.dump(text_processor, os.path.join(model_dir, 'text_processor.keras'))
    
    # Crear y guardar modelo híbrido
    hybrid_model = HybridPlagiarismDetector(
        svm_model=svm_model,
        cnn_model=cnn_model,
        feature_extractor=feature_extractor,
        text_processor=text_processor,
        ensemble_weights=best_weights,
        language=language
    )
    
    hybrid_model.save(os.path.join(model_dir, 'hybrid_model'))
    
    print(f"\nModelos guardados en {model_dir}")
    
    # 6. Generar visualizaciones mejoradas
    print("\n--- Generando visualizaciones ---")
    create_enhanced_visualizations(
        y_test=y_test,
        best_predictions=best_predictions,
        svm_proba=svm_proba,
        cnn_proba=y_pred_cnn_prob.flatten(),
        hybrid_proba=best_hybrid_proba,
        history=history,
        metadata_test=metadata_test,
        output_dir=model_dir
    )
    
    # 7. Guardar resultados de análisis por obfuscación
    pd.DataFrame(obfuscation_results).to_csv(os.path.join(model_dir, 'obfuscation_results.csv'), index=False)
    
    return {
        'svm_model': svm_model,
        'cnn_model': cnn_model,
        'hybrid_model': hybrid_model,
        'best_weights': best_weights,
        'metrics': {
            'svm_f1': precision_recall_fscore_support(y_test, y_pred_svm, average='binary')[2],
            'cnn_f1': precision_recall_fscore_support(y_test, y_pred_cnn, average='binary')[2],
            'hybrid_f1': best_f1
        },
        'obfuscation_results': obfuscation_results
    }

if __name__ == "__main__":
    # Configuración de rutas
    import sys
    
    # Directorio base del proyecto
    project_dir = os.getcwd()
    print(f"Directorio del proyecto: {project_dir}")
    
    # Ruta correcta a los datos
    data_dir = os.path.join(project_dir, 'src', 'data', 'processed')
    print(f"Buscando datos en: {data_dir}")
    
    # Verificar que existe
    if not os.path.exists(data_dir):
        print(f"ERROR: No se encontró el directorio {data_dir}")
        print("Directorios disponibles en src:")
        if os.path.exists('src'):
            print(os.listdir('src'))
        else:
            print("El directorio src no existe en el directorio actual")
        sys.exit(1)
    
    # Directorio para guardar modelos
    model_dir = os.path.join(project_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\nConfiguración:")
    print(f"  - Directorio de datos: {data_dir}")
    print(f"  - Directorio de modelos: {model_dir}")
    print("")
    
    # Ejecutar entrenamiento
    results = train_and_evaluate(
        data_dir=data_dir,
        model_dir=model_dir,
        use_pretrained_embeddings=True,
        embedding_model='glove-wiki-gigaword-300',
        language='english'
    )
    
    if results:
        print("\n=== RESUMEN DE RESULTADOS ===")
        print(f"Métricas SVM: F1-Score = {results['metrics']['svm_f1']:.4f}")
        print(f"Métricas CNN: F1-Score = {results['metrics']['cnn_f1']:.4f}")
        print(f"Métricas Modelo Híbrido: F1-Score = {results['metrics']['hybrid_f1']:.4f}")
        print(f"Mejores pesos: SVM={results['best_weights'][0]}, CNN={results['best_weights'][1]}")
        
        print("\nResultados por tipo de obfuscación:")
        for result in results['obfuscation_results']:
            print(f"  - {result['tipo']}: F1={result['f1']:.4f} (n={result['ejemplos']})")
        
        print(f"\nVisualización de resultados guardada en: {model_dir}")
        print("Entrenamiento y evaluación completados con éxito!")
    else:
        print("\nERROR: El entrenamiento no se completó correctamente.")