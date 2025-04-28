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

def train_and_evaluate(data_dir='data/processed', model_dir='models'):
    """
    Entrena y evalúa los modelos con el dataset generado.
    
    Args:
        data_dir: Directorio con los datos procesados
        model_dir: Directorio donde guardar los modelos
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
    
    cnn_model = PlagiarismCNN(
        vocab_size=vocab_size,
        embedding_dim=100,
        max_length=max_length,
        filters_per_size=64,
        filter_sizes=[3, 4, 5],
        dropout_rate=0.5
    )
    
    # Definir callbacks para CNN
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Entrenar CNN
    history = cnn_model.fit(
        X_cnn_source_train, 
        X_cnn_suspicious_train, 
        y_train,
        validation_split=0.1,
        batch_size=32,
        epochs=20,
        callbacks=[early_stopping]
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
    
    # Crear y guardar modelo híbrido
    hybrid_model = HybridPlagiarismDetector(
        svm_model=svm_model,
        cnn_model=cnn_model,
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
    
    return {
        'svm_model': svm_model,
        'cnn_model': cnn_model,
        'hybrid_model': hybrid_model,
        'best_weights': best_weights,
        'metrics': {
            'svm_f1': precision_recall_fscore_support(y_test, y_pred_svm, average='binary')[2],
            'cnn_f1': precision_recall_fscore_support(y_test, y_pred_cnn, average='binary')[2],
            'hybrid_f1': best_f1
        }
    }

if __name__ == "__main__":
    results = train_and_evaluate()
    print("\nEntrenamiento y evaluación completados!")
    print(f"F1-Score SVM: {results['metrics']['svm_f1']:.4f}")
    print(f"F1-Score CNN: {results['metrics']['cnn_f1']:.4f}")
    print(f"F1-Score Híbrido: {results['metrics']['hybrid_f1']:.4f}")