import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

from models.svm_model import PlagiarismSVM
from models.cnn_model import PlagiarismCNN
from integration.hybrid_model import HybridPlagiarismDetector
from features.feature_extractor import FeatureExtractor

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
    print("Cargando datos...")
    X_svm_train = np.load(os.path.join(data_dir, 'X_svm_train.npy'))
    X_cnn_source_train = np.load(os.path.join(data_dir, 'X_cnn_source_train.npy'))
    X_cnn_suspicious_train = np.load(os.path.join(data_dir, 'X_cnn_suspicious_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    
    X_svm_test = np.load(os.path.join(data_dir, 'X_svm_test.npy'))
    X_cnn_source_test = np.load(os.path.join(data_dir, 'X_cnn_source_test.npy'))
    X_cnn_suspicious_test = np.load(os.path.join(data_dir, 'X_cnn_suspicious_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    metadata_test = pd.read_csv(os.path.join(data_dir, 'metadata_test.csv'))
    
    print(f"Datos cargados: {len(y_train)} ejemplos de entrenamiento, {len(y_test)} de prueba")
    
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
    for obfuscation_type in metadata_test['obfuscation'].unique():
        mask = metadata_test['obfuscation'] == obfuscation_type
        if sum(mask) == 0:
            continue
            
        y_true_subset = metadata_test.loc[mask, 'is_plagiarism'].values
        y_pred_subset = metadata_test.loc[mask, 'hybrid_pred'].values
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_subset, y_pred_subset, average='binary', zero_division=0)
        
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
    
    # Crear y guardar modelo híbrido
    hybrid_model = HybridPlagiarismDetector(
        svm_model=svm_model,
        cnn_model=cnn_model,
        feature_extractor=feature_extractor,
        ensemble_weights=best_weights
    )
    
    hybrid_model.save(os.path.join(model_dir, 'hybrid_model'))
    
    print(f"\nModelos guardados en {model_dir}")
    
    # 6. Visualizar resultados
    plt.figure(figsize=(12, 5))
    
    # Matriz de confusión para el modelo híbrido
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, best_predictions)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de confusión (Híbrido)')
    plt.colorbar()
    plt.xticks([0, 1], ['No plagio', 'Plagio'])
    plt.yticks([0, 1], ['No plagio', 'Plagio'])
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    
    # Añadir valores numéricos
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    # Curvas históricas de la CNN si hay historia
    if hasattr(history, 'history'):
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='train_accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.title('Precisión durante entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'results.png'))
    
    # 7. Guardar resultados de análisis por obfuscación
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
    results = train_and_evaluate(
        use_pretrained_embeddings=True,
        embedding_model='glove-wiki-gigaword-300',
        language='english'
    )
    print("\nEntrenamiento y evaluación completados!")
    print(f"F1-Score SVM: {results['metrics']['svm_f1']:.4f}")
    print(f"F1-Score CNN: {results['metrics']['cnn_f1']:.4f}")
    print(f"F1-Score Híbrido: {results['metrics']['hybrid_f1']:.4f}")