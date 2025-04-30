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
        Inicializa el extractor de características para modelos SVM y CNN.

        Args:
            n_gram_ranges: Lista de tuplas que define el tamaño de n-gramas a usar para TF-IDF.
                Por ejemplo, (1, 1) genera unigramas, (2, 3) genera bigramas y trigramas.
                Esto sirve para capturar combinaciones de palabras que pueden ser indicativas de plagio o similitud textual.
            max_features: Número máximo de términos que se conservarán por vectorizador.
            language: Idioma usado para eliminar palabras vacías en TF-IDF (como 'y', 'el', 'de').
            use_pretrained_embeddings: Si se deben usar vectores de palabras preentrenados (como GloVe).
            embedding_model: Nombre del modelo de embeddings a utilizar.
            embedding_dim: Dimensión de los vectores de palabras (por ejemplo, 300).
            max_sequence_length: Longitud máxima de secuencia (en palabras) para las entradas a la CNN.
        """
        self.n_gram_ranges = n_gram_ranges
        self.max_features = max_features
        self.language = language
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length

        # Inicializa el vocabulario con dos tokens especiales:
        # <PAD> se usa para completar secuencias más cortas
        # <UNK> representa palabras desconocidas
        self.token_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_token = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2

        # Cargar los vectores de palabras preentrenados
        self.word_vectors = None
        if self.use_pretrained_embeddings:
            self._load_word_vectors()

        # Crea un vectorizador TF-IDF por cada rango de n-gramas solicitado.
        # Esto permite capturar similitudes en distintas granularidades: palabras sueltas, pares, frases cortas, etc.
        # Para que lo entiendan mejor:
        # Qué tan frecuente es la palabra en este documento específico (Term Frequency)
        # Qué tan rara es esta palabra en general (Inverse Document Frequency)
        
        self.vectorizers = {}
        for i, (min_n, max_n) in enumerate(n_gram_ranges):
            self.vectorizers[f'tfidf_{min_n}_{max_n}'] = TfidfVectorizer(
                ngram_range=(min_n, max_n),
                max_features=max_features,
                analyzer='word',
                stop_words=self.language
            )

    def _load_word_vectors(self):
        """
        Carga los vectores de palabras (embeddings) preentrenados desde Gensim.
        Estos vectores permiten representar cada palabra como un vector numérico continuo.
        """
        try:
            print(f"Cargando modelo de embeddings: {self.embedding_model}")
            self.word_vectors = gensim_downloader.load(self.embedding_model)
            print(f"Modelo cargado: {len(self.word_vectors.index_to_key)} palabras")
        except Exception as e:
            print(f"Error al cargar embeddings: {e}")
            print("Continuando sin embeddings preentrenados")
            self.use_pretrained_embeddings = False
            self.word_vectors = None

    def extract_svm_features(self, source_text: str, suspicious_text: str) -> np.ndarray:
        """
        Extrae un vector de características numéricas útiles para entrenar un SVM.
        Este vector combina métricas de similitud léxica, medidas TF-IDF, relación de longitudes
        y diversidad léxica (TTR).
        """
        features = []

        # Similitud léxica: mide qué tan similares son los textos en términos de palabras únicas
        features.append(self._jaccard_similarity(source_text, suspicious_text))
        features.append(self._containment_measure(source_text, suspicious_text))

        # Similitud semántica usando vectores TF-IDF y similitud del coseno
        for name, vectorizer in self.vectorizers.items():
            similarity = self._tfidf_cosine_similarity(vectorizer, source_text, suspicious_text)
            features.append(similarity)

        # Comparación de longitudes: útil para detectar fragmentos insertados o resumidos
        features.append(len(source_text) / (len(suspicious_text) + 1))

        # Diversidad léxica: proporción de palabras únicas respecto al total
        source_tokens = nltk.word_tokenize(source_text.lower())
        suspicious_tokens = nltk.word_tokenize(suspicious_text.lower())
        source_ttr = len(set(source_tokens)) / (len(source_tokens) + 1)
        suspicious_ttr = len(set(suspicious_tokens)) / (len(suspicious_tokens) + 1)

        # Incluye ambos TTRs y su diferencia absoluta
        features.extend([source_ttr, suspicious_ttr, abs(source_ttr - suspicious_ttr)])

        return np.array(features)

    def prepare_cnn_input(self, source_text: str, suspicious_text: str, update_vocab: bool = True) -> Dict:
        """
        Convierte textos en secuencias de índices numéricos, con padding a longitud fija.
        Estas secuencias pueden ser usadas como entrada para un modelo CNN.
        """
        # Tokeniza y recorta ambos textos a la longitud máxima permitida
        source_tokens = nltk.word_tokenize(source_text.lower())[:self.max_sequence_length]
        suspicious_tokens = nltk.word_tokenize(suspicious_text.lower())[:self.max_sequence_length]

        # Solo durante entrenamiento, actualiza el vocabulario con nuevas palabras
        if update_vocab:
            self._update_vocabulary(source_tokens + suspicious_tokens)

        # Convierte tokens en sus respectivos índices según el vocabulario
        source_indices = [self._token_to_index(token) for token in source_tokens]
        suspicious_indices = [self._token_to_index(token) for token in suspicious_tokens]

        # Padding con ceros hasta que las secuencias tengan la misma longitud
        source_indices += [0] * (self.max_sequence_length - len(source_indices))
        suspicious_indices += [0] * (self.max_sequence_length - len(suspicious_indices))

        return {
            'source_sequence': np.array(source_indices),
            'suspicious_sequence': np.array(suspicious_indices),
            'vocab_size': self.vocab_size,
            'token_to_idx': self.token_to_idx
        }

    def _update_vocabulary(self, tokens: List[str]):
        """
        Agrega tokens nuevos al vocabulario. Cada token recibe un índice numérico único.
        Esto se usa para mapear palabras a posiciones en la matriz de embeddings.
        """
        for token in tokens:
            if token not in self.token_to_idx:
                self.token_to_idx[token] = self.vocab_size
                self.idx_to_token[self.vocab_size] = token
                self.vocab_size += 1

    def _token_to_index(self, token: str) -> int:
        """
        Devuelve el índice numérico de un token.
        Si el token no está en el vocabulario, devuelve 1 (que representa <UNK>).
        """
        return self.token_to_idx.get(token, 1)

    def extract_features_for_fragment_pair(self, source_fragment: str, suspicious_fragment: str, update_vocab: bool = True) -> Dict:
        """
        Llama a los métodos de SVM y CNN para obtener todas las características del par de textos.
        Útil para entrenamiento o predicción combinada.
        """
        svm_features = self.extract_svm_features(source_fragment, suspicious_fragment)
        cnn_input = self.prepare_cnn_input(source_fragment, suspicious_fragment, update_vocab)

        return {
            'svm_features': svm_features,
            'cnn_input': cnn_input
        }

    def create_embedding_matrix(self) -> np.ndarray:
        """
        Crea una matriz que asocia cada índice del vocabulario con un vector (embedding).
        Si el modelo preentrenado no contiene la palabra, se asigna un vector aleatorio.
        El índice 0 (<PAD>) se representa con ceros.
        """
        if not self.use_pretrained_embeddings or self.word_vectors is None:
            print("Creando matriz de embeddings aleatoria")
            embedding_matrix = np.random.uniform(-0.25, 0.25, (self.vocab_size, self.embedding_dim))
            embedding_matrix[0] = np.zeros(self.embedding_dim)
            return embedding_matrix

        print(f"Creando matriz de embeddings para {self.vocab_size} tokens")
        embedding_matrix = np.random.uniform(-0.25, 0.25, (self.vocab_size, self.embedding_dim))
        embedding_matrix[0] = np.zeros(self.embedding_dim)

        found_embeddings = 0
        for token, idx in self.token_to_idx.items():
            if token in self.word_vectors:
                embedding_matrix[idx] = self.word_vectors[token]
                found_embeddings += 1

        print(f"Embeddings encontrados: {found_embeddings}/{self.vocab_size} ({found_embeddings/self.vocab_size*100:.2f}%)")
        return embedding_matrix

    def save_vocabulary(self, filepath: str):
        """
        Guarda en disco el vocabulario actual (tokens e índices) usando pickle.
        Esto permite reutilizar el mismo vocabulario durante inferencia.
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
        Carga un vocabulario previamente guardado.
        Útil para mantener consistencia entre entrenamiento y evaluación.
        """
        with open(filepath, 'rb') as f:
            vocabulary_data = pickle.load(f)

        self.token_to_idx = vocabulary_data['token_to_idx']
        self.idx_to_token = vocabulary_data['idx_to_token']
        self.vocab_size = vocabulary_data['vocab_size']

        print(f"Vocabulario cargado desde {filepath}: {self.vocab_size} tokens")

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula la similitud de Jaccard entre dos textos:
        |A ∩ B| / |A u B|, donde A y B son conjuntos de tokens.
        """
        tokens1 = set(nltk.word_tokenize(text1.lower()))
        tokens2 = set(nltk.word_tokenize(text2.lower()))
        return len(tokens1 & tokens2) / (len(tokens1 | tokens2) + 1e-10)

    def _containment_measure(self, text1: str, text2: str, n: int = 3) -> float:
        """
        Calcula la proporción de n-gramas de text1 que están presentes en text2.
        Es útil para detectar cuánto contenido exacto fue copiado.
        """
        def get_ngrams(text, n):
            tokens = nltk.word_tokenize(text.lower())
            return set(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))

        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        return len(ngrams1 & ngrams2) / (len(ngrams1) + 1e-10)

    def _tfidf_cosine_similarity(self, vectorizer, text1: str, text2: str) -> float:
        """
        Convierte los textos en vectores TF-IDF y calcula la similitud del coseno entre ellos.
        Esto captura similitud semántica y contextual basada en términos frecuentes e importantes.
        """
        texts = [text1, text2]
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            return 0.0