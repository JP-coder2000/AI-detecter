import numpy as np
from typing import Dict, List, Tuple, Any, Union
import joblib
import os

from src.models.svm_model import PlagiarismSVM
from src.models.cnn_model import PlagiarismCNN
from src.features.feature_extractor import FeatureExtractor

class HybridPlagiarismDetector:
    def __init__(self, svm_model: PlagiarismSVM = None, 
                 cnn_model: PlagiarismCNN = None,
                 feature_extractor: FeatureExtractor = None,
                 ensemble_weights: Tuple[float, float] = (0.5, 0.5)):
        """
        Inicializa el detector híbrido de plagio.
        
        Args:
            svm_model: Modelo SVM preentrenado
            cnn_model: Modelo CNN preentrenado
            feature_extractor: Extractor de características
            ensemble_weights: Pesos para combinar predicciones (SVM, CNN)
        """
        self.svm_model = svm_model
        self.cnn_model = cnn_model
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.ensemble_weights = ensemble_weights
    
    def predict(self, source_text: str, suspicious_text: str) -> Dict:
        """
        Predice si el texto sospechoso es plagio del texto fuente.
        
        Args:
            source_text: Texto del documento fuente
            suspicious_text: Texto del documento sospechoso
            
        Returns:
            Diccionario con resultados de predicción
        """
        # Extraer características
        features = self.feature_extractor.extract_features_for_fragment_pair(source_text, suspicious_text)
        
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
        clone_type = self._infer_clone_type(source_text, suspicious_text, ensemble_score)
        
        return {
            'is_plagiarism': bool(is_plagiarism),
            'confidence': float(ensemble_score),
            'svm_score': float(svm_prediction),
            'cnn_score': float(cnn_prediction),
            'clone_type': clone_type
        }
    
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
        joblib.dump(self.svm_model, os.path.join(directory, 'svm_model.pkl'))
        
        # Guardar CNN
        self.cnn_model.save(os.path.join(directory, 'cnn_model'))
        
        # Guardar extractor de características
        joblib.dump(self.feature_extractor, os.path.join(directory, 'feature_extractor.pkl'))
        
        # Guardar pesos de ensamble
        joblib.dump(self.ensemble_weights, os.path.join(directory, 'ensemble_weights.pkl'))
    
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
        svm_model = joblib.load(os.path.join(directory, 'svm_model.pkl'))
        
        # Cargar CNN
        cnn_model = PlagiarismCNN.load(os.path.join(directory, 'cnn_model'))
        
        # Cargar extractor de características
        feature_extractor = joblib.load(os.path.join(directory, 'feature_extractor.pkl'))
        
        # Cargar pesos de ensamble
        ensemble_weights = joblib.load(os.path.join(directory, 'ensemble_weights.pkl'))
        
        return cls(
            svm_model=svm_model,
            cnn_model=cnn_model,
            feature_extractor=feature_extractor,
            ensemble_weights=ensemble_weights
        )