import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Union
import joblib
import os
import seaborn as sns
from models.svm_model import PlagiarismSVM
from models.cnn_model import PlagiarismCNN
from features.feature_extractor import FeatureExtractor
from preprocessing.text_processor import TextProcessor

class HybridPlagiarismDetector:
    def __init__(self, svm_model: PlagiarismSVM = None, 
                 cnn_model: PlagiarismCNN = None,
                 feature_extractor: FeatureExtractor = None,
                 text_processor: TextProcessor = None,
                 ensemble_weights: Tuple[float, float] = (0.5, 0.5),
                 language: str = 'english'):
        """
        Inicializa el detector híbrido de plagio.
        
        Args:
            svm_model: Modelo SVM preentrenado
            cnn_model: Modelo CNN preentrenado
            feature_extractor: Extractor de características
            text_processor: Procesador de texto para preprocesamiento
            ensemble_weights: Pesos para combinar predicciones (SVM, CNN)
            language: Idioma para procesamiento ('english', 'spanish', etc.)
        """
        self.svm_model = svm_model
        self.cnn_model = cnn_model
        self.feature_extractor = feature_extractor or FeatureExtractor(language=language)
        self.text_processor = text_processor or TextProcessor(language=language)
        self.ensemble_weights = ensemble_weights
        self.language = language
    
    def predict(self, source_text: str, suspicious_text: str, preprocess: bool = True) -> Dict:
        """
        Predice si el texto sospechoso es plagio del texto fuente.
        
        Args:
            source_text: Texto del documento fuente
            suspicious_text: Texto del documento sospechoso
            preprocess: Si se debe preprocesar el texto
            
        Returns:
            Diccionario con resultados de predicción
        """
        # Preprocesar texto si se solicita
        if preprocess:
            source_processed = self.text_processor.preprocess_text(source_text)['processed']
            suspicious_processed = self.text_processor.preprocess_text(suspicious_text)['processed']
        else:
            source_processed = source_text
            suspicious_processed = suspicious_text
        
        # Extraer características
        features = self.feature_extractor.extract_features_for_fragment_pair(
            source_processed, suspicious_processed, update_vocab=False)
        
        # Predicción SVM
        svm_features = features['svm_features'].reshape(1, -1)
        svm_prediction = self.svm_model.predict_proba(svm_features)[0, 1]  # Probabilidad de clase positiva
        
        # Predicción CNN
        cnn_input = features['cnn_input']
        source_sequence = cnn_input['source_sequence'].reshape(1, -1)
        suspicious_sequence = cnn_input['suspicious_sequence'].reshape(1, -1)
        cnn_prediction = self.cnn_model.predict(source_sequence, suspicious_sequence)[0, 0]
        
        # Combinación ponderada
        w_svm, w_cnn = self.ensemble_weights
        ensemble_score = w_svm * svm_prediction + w_cnn * cnn_prediction
        
        # Determinar clase y nivel de confianza
        is_plagiarism = ensemble_score >= 0.5
        
        # Inferir tipo de clon (I, II o III)
        clone_type = self._infer_clone_type(source_processed, suspicious_processed, ensemble_score)
        
        return {
            'is_plagiarism': bool(is_plagiarism),
            'confidence': float(ensemble_score),
            'svm_score': float(svm_prediction),
            'cnn_score': float(cnn_prediction),
            'clone_type': clone_type,
            # Incluir fragmentos originales y procesados para referencia
            'fragments': {
                'source_original': source_text[:500] + ('...' if len(source_text) > 500 else ''),
                'suspicious_original': suspicious_text[:500] + ('...' if len(suspicious_text) > 500 else ''),
                'source_processed': source_processed[:500] + ('...' if len(source_processed) > 500 else ''),
                'suspicious_processed': suspicious_processed[:500] + ('...' if len(suspicious_processed) > 500 else '')
            }
        }
    
    def process_document_pair(self, source_doc: str, suspicious_doc: str, 
                             window_size: int = 500, stride: int = 250) -> Dict:
        """
        Procesa un par de documentos completos, usando una ventana deslizante.
        
        Args:
            source_doc: Texto completo del documento fuente
            suspicious_doc: Texto completo del documento sospechoso
            window_size: Tamaño de la ventana de análisis
            stride: Desplazamiento entre ventanas consecutivas
            
        Returns:
            Resultados de la detección de plagio para cada ventana
        """
        results = []
        
        # Preprocesar documentos completos
        source_processed = self.text_processor.preprocess_text(source_doc)
        suspicious_processed = self.text_processor.preprocess_text(suspicious_doc)
        
        # Tokenizar en oraciones para mantener integridad semántica
        source_sents = source_processed['sentences']
        suspicious_sents = suspicious_processed['sentences']
        
        # Generar ventanas de texto
        source_windows = self._create_text_windows(source_sents, window_size, stride)
        suspicious_windows = self._create_text_windows(suspicious_sents, window_size, stride)
        
        print(f"Generadas {len(source_windows)} ventanas para documento fuente")
        print(f"Generadas {len(suspicious_windows)} ventanas para documento sospechoso")
        
        # Matriz de similitud para todas las combinaciones posibles
        similarity_matrix = np.zeros((len(source_windows), len(suspicious_windows)))
        clone_types = {}
        
        # Analizar cada par de ventanas
        for i, source_window in enumerate(source_windows):
            for j, suspicious_window in enumerate(suspicious_windows):
                # Procesar este par de fragmentos
                prediction = self.predict(source_window, suspicious_window, preprocess=False)
                
                # Guardar score y tipo de clon
                similarity_matrix[i, j] = prediction['confidence']
                
                # Si es plagio, guardar información detallada
                if prediction['is_plagiarism']:
                    key = f"{i}_{j}"
                    clone_types[key] = prediction['clone_type']
                    
                    results.append({
                        'source_window_idx': i,
                        'suspicious_window_idx': j,
                        'source_window': source_window,
                        'suspicious_window': suspicious_window,
                        'confidence': prediction['confidence'],
                        'clone_type': prediction['clone_type'],
                        'svm_score': prediction['svm_score'],
                        'cnn_score': prediction['cnn_score']
                    })
        
        # Ordenar resultados por confianza
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'detailed_results': results,
            'similarity_matrix': similarity_matrix,
            'clone_types': clone_types,
            'total_comparisons': len(source_windows) * len(suspicious_windows),
            'plagiarism_detected': len(results) > 0,
            'source_windows': len(source_windows),
            'suspicious_windows': len(suspicious_windows)
        }
    
    def _create_text_windows(self, sentences: List[str], window_size: int, stride: int) -> List[str]:
        """
        Crea ventanas de texto a partir de oraciones.
        
        Args:
            sentences: Lista de oraciones
            window_size: Tamaño aproximado de la ventana en caracteres
            stride: Superposición entre ventanas consecutivas
            
        Returns:
            Lista de fragmentos de texto (ventanas)
        """
        windows = []
        current_window = ""
        current_size = 0
        
        for sentence in sentences:
            # Añadir oración a la ventana actual
            if current_size + len(sentence) <= window_size:
                current_window += sentence + " "
                current_size += len(sentence) + 1
            else:
                # Si la ventana está llena, guardarla y crear nueva ventana
                if current_window:
                    windows.append(current_window.strip())
                
                # Avanzar la ventana según el stride
                chars_to_keep = max(0, current_size - stride)
                if chars_to_keep > 0:
                    # Mantener palabras completas
                    words = current_window.split()
                    current_window = ""
                    current_size = 0
                    
                    # Reconstruir ventana manteniendo palabras para alcanzar el tamaño deseado
                    for word in reversed(words):
                        if current_size + len(word) + 1 <= chars_to_keep:
                            current_window = word + " " + current_window
                            current_size += len(word) + 1
                        else:
                            break
                else:
                    current_window = ""
                    current_size = 0
                
                # Añadir la oración actual a la nueva ventana
                current_window += sentence + " "
                current_size += len(sentence) + 1
        
        # Añadir última ventana si no está vacía
        if current_window:
            windows.append(current_window.strip())
        
        return windows
    
    def _infer_clone_type(self, source_text: str, suspicious_text: str, score: float) -> str:
        """
        Infiere el tipo de clon (I, II o III) basado en características textuales.
        
        Args:
            source_text: Texto del documento fuente
            suspicious_text: Texto del documento sospechoso
            score: Puntuación de plagio
            
        Returns:
            Tipo de clon inferido
        """
        # Calcular similitud léxica exacta (para distinguir principalmente tipo I)
        jaccard_sim = self.feature_extractor._jaccard_similarity(source_text, suspicious_text)
        
        # Umbral para tipos de clones (refinados experimentalmente)
        if jaccard_sim > 0.8 and score > 0.9:
            return "Type I"  # Clones casi idénticos
        elif jaccard_sim > 0.5 and score > 0.7:
            return "Type II"  # Cambios en identificadores/valores
        elif score > 0.5:
            return "Type III"  # Paráfrasis más complejas
        else:
            return "Not a clone"
    
    def save(self, directory: str):
        """
        Guarda los componentes del modelo híbrido.
        
        Args:
            directory: Directorio donde guardar los modelos
        """
        os.makedirs(directory, exist_ok=True)
        
        # Guardar SVM
        joblib.dump(self.svm_model, os.path.join(directory, 'svm_model.keras'))
        
        # Guardar CNN
        self.cnn_model.save(os.path.join(directory, 'cnn_model.keras'))
        
        # Guardar extractor de características
        joblib.dump(self.feature_extractor, os.path.join(directory, 'feature_extractor.keras'))
        
        # Guardar procesador de texto
        joblib.dump(self.text_processor, os.path.join(directory, 'text_processor.keras'))
        
        # Guardar pesos de ensamble y configuración
        config = {
            'ensemble_weights': self.ensemble_weights,
            'language': self.language
        }
        joblib.dump(config, os.path.join(directory, 'config.keras'))
        
        # Guardar vocabulario específicamente
        self.feature_extractor.save_vocabulary(os.path.join(directory, 'vocabulary.pkl'))
    
    @classmethod
    def load(cls, directory: str):
        """
        Carga un modelo híbrido guardado.
        
        Args:
            directory: Directorio donde se encuentran los modelos guardados
            
        Returns:
            Modelo híbrido cargado
        """
        # Cargar SVM
        svm_model = joblib.load(os.path.join(directory, 'svm_model.keras'))
        
        # Cargar CNN
        cnn_model = tf.keras.models.load_model(os.path.join(directory, 'cnn_model.keras'))
        
        # Cargar extractor de características
        feature_extractor = joblib.load(os.path.join(directory, 'feature_extractor.keras'))
        
        # Cargar procesador de texto
        text_processor = joblib.load(os.path.join(directory, 'text_processor.keras'))
        
        # Cargar configuración
        config = joblib.load(os.path.join(directory, 'config.keras'))
        
        # Cargar vocabulario específicamente si existe
        vocab_path = os.path.join(directory, 'vocabulary.pkl')
        if os.path.exists(vocab_path):
            feature_extractor.load_vocabulary(vocab_path)
        
        return cls(
            svm_model=svm_model,
            cnn_model=cnn_model,
            feature_extractor=feature_extractor,
            text_processor=text_processor,
            ensemble_weights=config['ensemble_weights'],
            language=config['language']
        )
    
    def visualize_results(self, results, output_path=None):
        """
        Visualiza los resultados de detección de plagio.
        
        Args:
            results: Resultados de process_document_pair
            output_path: Ruta donde guardar la visualización (opcional)
            
        Returns:
            Figura de matplotlib
        """
        # Crear figura principal
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Mapa de calor de similitud
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        sns.heatmap(results['similarity_matrix'], cmap='YlOrRd', 
                   xticklabels=10, yticklabels=10)
        ax1.set_title('Matriz de Similitud entre Fragmentos')
        ax1.set_xlabel('Fragmentos del Documento Sospechoso')
        ax1.set_ylabel('Fragmentos del Documento Fuente')
        
        # 2. Top fragmentos con mayor similitud
        top_results = results['detailed_results'][:min(5, len(results['detailed_results']))]
        
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        clone_types = [r['clone_type'] for r in top_results]
        confidence = [r['confidence'] for r in top_results]
        
        if top_results:
            indices = range(len(top_results))
            bars = ax2.bar(indices, confidence, color=['red' if t == 'Type I' else 
                                                     'orange' if t == 'Type II' else 
                                                     'yellow' for t in clone_types])
            
            # Añadir etiquetas
            for i, bar in enumerate(bars):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f"{clone_types[i]}", ha='center', va='bottom')
            
            ax2.set_title('Top 5 Fragmentos con Mayor Similitud')
            ax2.set_xlabel('Índice de Fragmento')
            ax2.set_ylabel('Confianza')
            ax2.set_ylim(0, 1.1)
            ax2.set_xticks(indices)
            ax2.set_xticklabels([f"{r['source_window_idx']}-{r['suspicious_window_idx']}" 
                                for r in top_results], rotation=45)
        else:
            ax2.text(0.5, 0.5, "No se detectó plagio", ha='center', va='center', fontsize=12)
            ax2.set_title('Resultados de Detección')
        
        # 3. Distribución de tipos de clones
        ax3 = plt.subplot2grid((2, 2), (1, 1))
        
        if results['clone_types']:
            clone_count = {'Type I': 0, 'Type II': 0, 'Type III': 0, 'Not a clone': 0}
            
            for clone_type in results['clone_types'].values():
                clone_count[clone_type] = clone_count.get(clone_type, 0) + 1
            
            types = list(clone_count.keys())
            counts = list(clone_count.values())
            
            colors = ['red', 'orange', 'yellow', 'green']
            wedges, texts, autotexts = ax3.pie(counts, labels=types, autopct='%1.1f%%',
                                             colors=colors, startangle=90)
            
            # Hacer las etiquetas más legibles
            for text in texts:
                text.set_fontsize(9)
            
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_weight('bold')
                
            ax3.set_title('Distribución de Tipos de Clones')
        else:
            ax3.text(0.5, 0.5, "No se detectó plagio", ha='center', va='center', fontsize=12)
            ax3.set_title('Distribución de Tipos de Clones')
            
        plt.tight_layout()
        
        # Guardar figura si se especifica una ruta
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualización guardada en {output_path}")
        
        return fig