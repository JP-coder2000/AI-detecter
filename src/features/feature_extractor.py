import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from typing import List, Dict, Tuple, Any, Union
import re

class FeatureExtractor:
    def __init__(self, n_gram_ranges: List[Tuple[int, int]] = [(1, 1), (2, 3)], 
                 max_features: int = 10000):
        """
        Inicializa el extractor de características.
        
        Args:
            n_gram_ranges: Lista de rangos de n-gramas (min, max) para diferentes vectorizadores
            max_features: Número máximo de características para TF-IDF
        """
        self.n_gram_ranges = n_gram_ranges
        self.max_features = max_features
        self.vectorizers = {}
        
        # Inicializar vectorizadores TF-IDF para diferentes rangos de n-gramas
        for i, (min_n, max_n) in enumerate(n_gram_ranges):
            self.vectorizers[f'tfidf_{min_n}_{max_n}'] = TfidfVectorizer(
                ngram_range=(min_n, max_n),
                max_features=max_features,
                analyzer='word',
                stop_words='english'
            )
    
    def extract_svm_features(self, source_text: str, suspicious_text: str) -> np.ndarray:
        """
        Extrae características para el modelo SVM, basado en medidas de similitud.
        
        Args:
            source_text: Texto del documento fuente
            suspicious_text: Texto del documento sospechoso
            
        Returns:
            Vector de características para SVM
        """
        features = []
        
        # Características de similitud léxica
        features.append(self._jaccard_similarity(source_text, suspicious_text))
        features.append(self._containment_measure(source_text, suspicious_text))
        
        # Características basadas en TF-IDF y similitud del coseno
        for name, vectorizer in self.vectorizers.items():
            similarity = self._tfidf_cosine_similarity(vectorizer, source_text, suspicious_text)
            features.append(similarity)
        
        # Características de longitud y proporción
        features.append(len(source_text) / (len(suspicious_text) + 1))  # +1 para evitar división por cero
        
        # Características de diversidad léxica (relación tipos/tokens)
        source_tokens = nltk.word_tokenize(source_text.lower())
        suspicious_tokens = nltk.word_tokenize(suspicious_text.lower())
        
        source_ttr = len(set(source_tokens)) / (len(source_tokens) + 1)  # Type-Token Ratio
        suspicious_ttr = len(set(suspicious_tokens)) / (len(suspicious_tokens) + 1)
        
        features.append(source_ttr)
        features.append(suspicious_ttr)
        features.append(abs(source_ttr - suspicious_ttr))  # Diferencia absoluta entre TTRs
        
        return np.array(features)
    
    def prepare_cnn_input(self, source_text: str, suspicious_text: str, max_length: int = 500) -> Dict:
        """
        Prepara entradas para el modelo CNN.
        
        Args:
            source_text: Texto del documento fuente
            suspicious_text: Texto del documento sospechoso
            max_length: Longitud máxima de secuencia
            
        Returns:
            Diccionario con datos de entrada para CNN
        """
        # Tokenizar textos
        source_tokens = nltk.word_tokenize(source_text.lower())[:max_length]
        suspicious_tokens = nltk.word_tokenize(suspicious_text.lower())[:max_length]
        
        # Crear un vocabulario sencillo (en producción usaríamos word embeddings preentrenados)
        all_tokens = set(source_tokens + suspicious_tokens)
        token_to_idx = {token: idx + 1 for idx, token in enumerate(all_tokens)}  # +1 para reservar 0 para padding
        
        # Convertir tokens a índices
        source_indices = [token_to_idx.get(token, 0) for token in source_tokens]
        suspicious_indices = [token_to_idx.get(token, 0) for token in suspicious_tokens]
        
        # Padding para longitud fija
        source_indices = source_indices + [0] * (max_length - len(source_indices))
        suspicious_indices = suspicious_indices + [0] * (max_length - len(suspicious_indices))
        
        return {
            'source_sequence': np.array(source_indices),
            'suspicious_sequence': np.array(suspicious_indices),
            'vocab_size': len(token_to_idx) + 1,  # +1 para el token de padding
            'token_to_idx': token_to_idx
        }
    
    def extract_features_for_fragment_pair(self, source_fragment: str, suspicious_fragment: str) -> Dict:
        """
        Extrae características para un par de fragmentos de texto.
        
        Args:
            source_fragment: Fragmento del documento fuente
            suspicious_fragment: Fragmento del documento sospechoso
            
        Returns:
            Diccionario con características SVM y datos para CNN
        """
        svm_features = self.extract_svm_features(source_fragment, suspicious_fragment)
        cnn_input = self.prepare_cnn_input(source_fragment, suspicious_fragment)
        
        return {
            'svm_features': svm_features,
            'cnn_input': cnn_input
        }
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calcula la similitud de Jaccard entre dos textos."""
        tokens1 = set(nltk.word_tokenize(text1.lower()))
        tokens2 = set(nltk.word_tokenize(text2.lower()))
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / (len(union) + 1e-10)  # Evitar división por cero
    
    def _containment_measure(self, text1: str, text2: str, n: int = 3) -> float:
        """Calcula la medida de contención de n-gramas."""
        def get_ngrams(text, n):
            tokens = nltk.word_tokenize(text.lower())
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.append(ngram)
            return set(ngrams)
        
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        
        intersection = ngrams1.intersection(ngrams2)
        
        # Contención: S(A,B) = |A ∩ B| / |A|
        return len(intersection) / (len(ngrams1) + 1e-10)
    
    def _tfidf_cosine_similarity(self, vectorizer, text1: str, text2: str) -> float:
        """Calcula la similitud del coseno entre dos textos utilizando TF-IDF."""
        texts = [text1, text2]
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            return 0.0  # En caso de error, devolver similitud cero