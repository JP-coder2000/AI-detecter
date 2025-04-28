import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Concatenate, Flatten
from tensorflow.keras.regularizers import l2
from typing import Dict, List, Tuple, Any, Union

class PlagiarismCNN:
    def __init__(self, vocab_size: int, 
                 embedding_dim: int = 100,
                 max_length: int = 500,
                 filters_per_size: int = 64,
                 filter_sizes: List[int] = [3, 4, 5],
                 dropout_rate: float = 0.5):
        """
        Inicializa el modelo CNN para detección de plagio.
        
        Args:
            vocab_size: Tamaño del vocabulario
            embedding_dim: Dimensión de los embeddings
            max_length: Longitud máxima de secuencia
            filters_per_size: Número de filtros por tamaño
            filter_sizes: Lista de tamaños de filtro
            dropout_rate: Tasa de dropout
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.filters_per_size = filters_per_size
        self.filter_sizes = filter_sizes
        self.dropout_rate = dropout_rate
        
        # Construir modelo
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """
        Construye la arquitectura CNN para detección de plagio.
        
        Returns:
            Modelo de Keras
        """
        # Entradas para fragmentos fuente y sospechosos
        source_input = Input(shape=(self.max_length,), name='source_input')
        suspicious_input = Input(shape=(self.max_length,), name='suspicious_input')
        
        # Capa de embedding compartida
        embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            name='embedding'
        )
        
        # Embeddings para cada entrada
        source_embedding = embedding_layer(source_input)
        suspicious_embedding = embedding_layer(suspicious_input)
        
        # Aplicar convoluciones con diferentes tamaños de filtro a cada entrada
        source_convs = []
        suspicious_convs = []
        
        for filter_size in self.filter_sizes:
            # Convoluciones para texto fuente
            source_conv = Conv1D(
                filters=self.filters_per_size,
                kernel_size=filter_size,
                padding='valid',
                activation='relu',
                kernel_regularizer=l2(0.001),
                name=f'source_conv_{filter_size}'
            )(source_embedding)
            source_pool = MaxPooling1D(
                pool_size=self.max_length - filter_size + 1,
                name=f'source_pool_{filter_size}'
            )(source_conv)
            source_convs.append(source_pool)
            
            # Convoluciones para texto sospechoso
            suspicious_conv = Conv1D(
                filters=self.filters_per_size,
                kernel_size=filter_size,
                padding='valid',
                activation='relu',
                kernel_regularizer=l2(0.001),
                name=f'suspicious_conv_{filter_size}'
            )(suspicious_embedding)
            suspicious_pool = MaxPooling1D(
                pool_size=self.max_length - filter_size + 1,
                name=f'suspicious_pool_{filter_size}'
            )(suspicious_conv)
            suspicious_convs.append(suspicious_pool)
        
        # Concatenar resultados de convolución
        source_concat = Concatenate(axis=1, name='source_concat')(source_convs)
        suspicious_concat = Concatenate(axis=1, name='suspicious_concat')(suspicious_convs)
        
        # Aplanar
        source_flat = Flatten(name='source_flatten')(source_concat)
        suspicious_flat = Flatten(name='suspicious_flatten')(suspicious_concat)
        
        # Concatenar representaciones de ambos textos
        concat = Concatenate(axis=1, name='concat')([source_flat, suspicious_flat])
        
        # Capas densas
        dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='dense1')(concat)
        dropout1 = Dropout(self.dropout_rate, name='dropout1')(dense1)
        dense2 = Dense(64, activation='relu', kernel_regularizer=l2(0.001), name='dense2')(dropout1)
        dropout2 = Dropout(self.dropout_rate, name='dropout2')(dense2)
        
        # Capa de salida
        output = Dense(1, activation='sigmoid', name='output')(dropout2)
        
        # Crear modelo
        model = Model(inputs=[source_input, suspicious_input], outputs=output)
        
        # Compilar modelo
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def fit(self, source_sequences: np.ndarray, 
            suspicious_sequences: np.ndarray, 
            labels: np.ndarray,
            validation_split: float = 0.2,
            batch_size: int = 32,
            epochs: int = 10,
            callbacks: List = None):
        """
        Entrena el modelo CNN.
        
        Args:
            source_sequences: Secuencias de texto fuente
            suspicious_sequences: Secuencias de texto sospechoso
            labels: Etiquetas (1: plagio, 0: no plagio)
            validation_split: Proporción de datos para validación
            batch_size: Tamaño del lote
            epochs: Número de épocas
            callbacks: Lista de callbacks de Keras
        """
        history = self.model.fit(
            [source_sequences, suspicious_sequences],
            labels,
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, source_sequences: np.ndarray, suspicious_sequences: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            source_sequences: Secuencias de texto fuente
            suspicious_sequences: Secuencias de texto sospechoso
            
        Returns:
            Vector de probabilidades de plagio
        """
        return self.model.predict([source_sequences, suspicious_sequences])
    
    def save(self, filepath: str):
        """Guarda el modelo en disco."""
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str):
        """Carga un modelo guardado."""
        return tf.keras.models.load_model(filepath)