import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from typing import List, Dict, Tuple, Any, Union
import re
import os
import pickle
import gensim.downloader as gensim_downloader

class FeatureExtractor:
    def __init__(self, n_gram_ranges: List[Tuple[int, int]] = [(1, 1), (2, 3)], 
                 max_features: int = 10000,
                 language: str = 'english',
                 use_pretrained_embeddings: bool = True,
                 embedding_model: str = 'glove-wiki-gigaword-300',
                 embedding_dim: int = 300,
                 max_sequence_length: int = 500):
        """
        Inicializa el extractor de características.
        
        Args:
            n_gram_ranges: Lista de rangos de n-gramas (min, max) para diferentes vectorizadores
            max_features: Número máximo de características para TF-IDF
            language: Idioma para procesamiento ('english', 'spanish', etc.)
            use_pretrained_embeddings: Si se deben usar embeddings preentrenados
            embedding_model: Nombre del modelo de embeddings a usar
            embedding_dim: Dimensión de los embeddings
            max_sequence_length: Longitud máxima de secuencia para CNN
        """
        self.n_gram_ranges = n_gram_ranges
        self.max_features = max_features
        self.language = language
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        
        # Vocabulario global para mantener consistencia entre calls
        self.token_to_idx = {"<PAD>": 0, "<UNK>": 1}  # Empezamos con tokens especiales
        self.idx_to_token = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2  # Contamos tokens especiales
        
        # Cargar embedding preentrenado si se solicita
        self.word_vectors = None
        if self.use_pretrained_embeddings:
            self._load_word_vectors()
        
        # Inicializar vectorizadores TF-IDF para diferentes rangos de n-gramas
        self.vectorizers = {}
        for i, (min_n, max_n) in enumerate(n_gram_ranges):
            self.vectorizers[f'tfidf_{min_n}_{max_n}'] = TfidfVectorizer(
                ngram_range=(min_n, max_n),
                max_features=max_features,
                analyzer='word',
                stop_words=self.language
            )
    
    def _load_word_vectors(self):
        """Carga los vectores de palabra preentrenados."""
        try:
            print(f"Cargando modelo de embeddings: {self.embedding_model}")
            # Utilizar gensim para cargar vectores preentrenados
            self.word_vectors = gensim_downloader.load(self.embedding_model)
            print(f"Modelo cargado: {len(self.word_vectors.index_to_key)} palabras")
        except Exception as e:
            print(f"Error al cargar embeddings: {e}")
            print("Continuando sin embeddings preentrenados")
            self.use_pretrained_embeddings = False
            self.word_vectors = None
    
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
    
    def prepare_cnn_input(self, source_text: str, suspicious_text: str, update_vocab: bool = True) -> Dict:
        """
        Prepara entradas para el modelo CNN.
        
        Args:
            source_text: Texto del documento fuente
            suspicious_text: Texto del documento sospechoso
            update_vocab: Si se debe actualizar el vocabulario (True durante entrenamiento, False durante inferencia)
            
        Returns:
            Diccionario con datos de entrada para CNN
        """
        # Tokenizar textos
        source_tokens = nltk.word_tokenize(source_text.lower())[:self.max_sequence_length]
        suspicious_tokens = nltk.word_tokenize(suspicious_text.lower())[:self.max_sequence_length]
        
        # Actualizar vocabulario si es necesario (solo en entrenamiento)
        if update_vocab:
            self._update_vocabulary(source_tokens + suspicious_tokens)
        
        # Convertir tokens a índices
        source_indices = [self._token_to_index(token) for token in source_tokens]
        suspicious_indices = [self._token_to_index(token) for token in suspicious_tokens]
        
        # Padding para longitud fija
        source_indices = source_indices + [0] * (self.max_sequence_length - len(source_indices))
        suspicious_indices = suspicious_indices + [0] * (self.max_sequence_length - len(suspicious_indices))
        
        return {
            'source_sequence': np.array(source_indices),
            'suspicious_sequence': np.array(suspicious_indices),
            'vocab_size': self.vocab_size,
            'token_to_idx': self.token_to_idx
        }
    
    def _update_vocabulary(self, tokens: List[str]):
        """
        Actualiza el vocabulario con nuevos tokens.
        
        Args:
            tokens: Lista de tokens a añadir al vocabulario
        """
        for token in tokens:
            if token not in self.token_to_idx:
                self.token_to_idx[token] = self.vocab_size
                self.idx_to_token[self.vocab_size] = token
                self.vocab_size += 1
    
    def _token_to_index(self, token: str) -> int:
        """
        Convierte un token a su índice en el vocabulario.
        
        Args:
            token: Token a convertir
            
        Returns:
            Índice del token o 1 (UNK) si no está en el vocabulario
        """
        return self.token_to_idx.get(token, 1)  # 1 es <UNK>
    
    def extract_features_for_fragment_pair(self, source_fragment: str, suspicious_fragment: str, update_vocab: bool = True) -> Dict:
        """
        Extrae características para un par de fragmentos de texto.
        
        Args:
            source_fragment: Fragmento del documento fuente
            suspicious_fragment: Fragmento del documento sospechoso
            update_vocab: Si se debe actualizar el vocabulario
            
        Returns:
            Diccionario con características SVM y datos para CNN
        """
        svm_features = self.extract_svm_features(source_fragment, suspicious_fragment)
        cnn_input = self.prepare_cnn_input(source_fragment, suspicious_fragment, update_vocab)
        
        return {
            'svm_features': svm_features,
            'cnn_input': cnn_input
        }
    
    def create_embedding_matrix(self) -> np.ndarray:
        """
        Crea una matriz de embeddings para el vocabulario actual utilizando los embeddings preentrenados.
        
        Returns:
            Matriz de embeddings [vocab_size x embedding_dim]
        """
        if not self.use_pretrained_embeddings or self.word_vectors is None:
            # Si no hay embeddings preentrenados, devolver una matriz aleatoria
            print("Creando matriz de embeddings aleatoria")
            embedding_matrix = np.random.uniform(-0.25, 0.25, (self.vocab_size, self.embedding_dim))
            embedding_matrix[0] = np.zeros(self.embedding_dim)  # Padding token con ceros
            return embedding_matrix
        
        # Crear matriz de embeddings
        print(f"Creando matriz de embeddings para {self.vocab_size} tokens")
        embedding_matrix = np.random.uniform(-0.25, 0.25, (self.vocab_size, self.embedding_dim))
        embedding_matrix[0] = np.zeros(self.embedding_dim)  # Padding token con ceros
        
        # Llenar con embeddings preentrenados
        found_embeddings = 0
        for token, idx in self.token_to_idx.items():
            if token in self.word_vectors:
                embedding_matrix[idx] = self.word_vectors[token]
                found_embeddings += 1
        
        print(f"Embeddings encontrados: {found_embeddings}/{self.vocab_size} ({found_embeddings/self.vocab_size*100:.2f}%)")
        
        return embedding_matrix
    
    def save_vocabulary(self, filepath: str):
        """
        Guarda el vocabulario en disco.
        
        Args:
            filepath: Ruta donde guardar el vocabulario
        """
        vocabulary_data = {
            'token_to_idx': self.token_to_idx,
            'idx_to_token': self.idx_to_token,
            'vocab_size': self.vocab_size
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(vocabulary_data, f)
        
        print(f"Vocabulario guardado en {filepath}: {self.vocab_size} tokens")
    
    def load_vocabulary(self, filepath: str):
        """
        Carga el vocabulario desde disco.
        
        Args:
            filepath: Ruta desde donde cargar el vocabulario
        """
        with open(filepath, 'rb') as f:
            vocabulary_data = pickle.load(f)
        
        self.token_to_idx = vocabulary_data['token_to_idx']
        self.idx_to_token = vocabulary_data['idx_to_token']
        self.vocab_size = vocabulary_data['vocab_size']
        
        print(f"Vocabulario cargado desde {filepath}: {self.vocab_size} tokens")
    
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