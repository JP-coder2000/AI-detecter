import re
import string
import nltk
import ssl
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List, Dict, Any, Union

# Configuración para evitar problemas de SSL en algunos entornos (Me pasaba mucho a mi JP, encontré que así lo podía solucionar)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Descarga recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextProcessor:
    def __init__(self, language: str = 'english'):
        """
        Inicializa el procesador de texto.
        
        Args:
            language: Idioma para stopwords y lematización ('english', 'german', 'spanish')
        """
        self.language = language # Cargamos el idioma
        self.stemmer = PorterStemmer() # Este sirve para que en dado caso de que hayan palabras con raíces similares, como correr, corrida, correo, lo deje en corr. (Este lo tengo desactivado por defecto, solo lo implementé para ver que tal funcionaba el modelo)
        self.lemmatizer = WordNetLemmatizer() # Este sirve para volver las palabras que estan conjugadas a su raíz, como por ejemplo: corriendo -> correr, corrí -> correr, etc. (Este lo tengo activado por defecto)
        
        # Stopwords según el idioma
        self.stop_words = set()
        if language == 'english':
            self.stop_words = set(stopwords.words('english'))
        elif language == 'german':
            self.stop_words = set(stopwords.words('german'))
        elif language == 'spanish':
            self.stop_words = set(stopwords.words('spanish'))
    
    def preprocess_text(self, text: str, 
                       remove_stopwords: bool = True,
                       stem_words: bool = False,
                       lemmatize: bool = True) -> Dict[str, Any]:
        """
        Preprocesa el texto aplicando varias técnicas.
        
        Args:
            text: Texto a preprocesar
            remove_stopwords: Si se deben eliminar stopwords
            stem_words: Si se debe aplicar stemming
            lemmatize: Si se debe aplicar lematización
            
        Returns:
            Diccionario con el texto procesado y metadatos
        """
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar números y puntuación
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenizar en oraciones y palabras
        sentences = sent_tokenize(text)
        
        processed_sentences = []
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            
            # Eliminar stopwords si se solicita
            if remove_stopwords:
                tokens = [t for t in tokens if t not in self.stop_words]
            
            # Aplicar stemming si se solicita
            if stem_words:
                tokens = [self.stemmer.stem(t) for t in tokens]
            
            # Aplicar lematización si se solicita
            if lemmatize:
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            
            # Reconstruir oración
            processed_sentence = ' '.join(tokens)
            processed_sentences.append(processed_sentence)
        
        # Reconstruir texto completo
        processed_text = ' '.join(processed_sentences)
        
        return {
            'original': text,
            'processed': processed_text,
            'sentences': sentences,
            'processed_sentences': processed_sentences,
            'n_sentences': len(sentences)
        }