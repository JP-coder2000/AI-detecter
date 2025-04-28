import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from typing import List, Dict, Any, Union

class PlagiarismSVM:
    def __init__(self, kernel: str = 'rbf', 
                 class_weight: Union[Dict, str] = 'balanced',
                 C: float = 1.0,
                 gamma: str = 'scale'):
        """
        Inicializa el modelo SVM para detección de plagio.
        
        Args:
            kernel: Kernel SVM ('linear', 'rbf', 'poly', etc.)
            class_weight: Pesos de las clases para manejar desbalances
            C: Parámetro de regularización
            gamma: Parámetro de kernel
        """
        self.kernel = kernel
        self.class_weight = class_weight
        self.C = C
        self.gamma = gamma
        
        # Crear pipeline con escalado y SVM
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel=self.kernel,
                class_weight=self.class_weight,
                C=self.C,
                gamma=self.gamma,
                probability=True
            ))
        ])
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Entrena el modelo SVM.
        
        Args:
            X: Matriz de características
            y: Vector de etiquetas (1: plagio, 0: no plagio)
        """
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            X: Matriz de características
            
        Returns:
            Vector de predicciones (1: plagio, 0: no plagio)
        """
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula probabilidades de predicción.
        
        Args:
            X: Matriz de características
            
        Returns:
            Matriz de probabilidades por clase
        """
        return self.pipeline.predict_proba(X)
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                cv: int = 5) -> Dict:
        """
        Optimiza hiperparámetros mediante validación cruzada.
        
        Args:
            X: Matriz de características
            y: Vector de etiquetas
            cv: Número de divisiones para validación cruzada
            
        Returns:
            Diccionario con los mejores hiperparámetros
        """
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.01, 0.1],
            'svm__kernel': ['rbf', 'linear', 'poly']
        }
        
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Actualizar modelo con mejores parámetros
        self.pipeline = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }