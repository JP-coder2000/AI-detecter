import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
import argparse
import collections
import time
import gc

from preprocessing.corpus_loader import PanCorpusLoader
from preprocessing.text_processor import TextProcessor
from features.feature_extractor import FeatureExtractor

def create_dataset(corpus_path, max_examples=0, test_size=0.2, random_state=42,
                  language='english', use_pretrained_embeddings=True, 
                  embedding_model='glove-wiki-gigaword-300', batch_mode=False):
    """
    Crea un dataset para detección de plagio usando todos los ejemplos disponibles.
    
    Args:
        corpus_path: Ruta al corpus PAN
        max_examples: Número máximo de ejemplos a generar (0 para usar todos los disponibles)
        test_size: Proporción del dataset para prueba
        random_state: Semilla aleatoria para reproducibilidad
        language: Idioma del corpus ('english', 'spanish', etc.)
        use_pretrained_embeddings: Si se deben usar embeddings preentrenados
        embedding_model: Modelo de embeddings a utilizar
        batch_mode: Si se debe ejecutar en modo batch con menos información detallada
    """
    start_time = time.time()
    
    print(f"Inicializando componentes para idioma: {language}")
    processor = TextProcessor(language=language)
    extractor = FeatureExtractor(
        language=language,
        use_pretrained_embeddings=use_pretrained_embeddings,
        embedding_model=embedding_model
    )
    
    # PASO 1: Encontrar casos de plagio directamente de los XML
    print("Buscando casos de plagio en el corpus...")
    suspicious_path = os.path.join(corpus_path, "external-detection-corpus", "suspicious-document")
    
    plagiarism_cases = []
    source_docs_needed = set()
    suspicious_docs_needed = set()
    
    # Buscar en todas las carpetas part*
    for part_dir in sorted(os.listdir(suspicious_path)):
        if not part_dir.startswith("part"):
            continue
            
        part_path = os.path.join(suspicious_path, part_dir)
        print(f"Buscando en {part_dir}...")
        
        for filename in os.listdir(part_path):
            if not filename.endswith(".xml"):
                continue
                
            suspicious_id = filename[:-4]  # Quitar extensión .xml
            file_path = os.path.join(part_path, filename)
            
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                for feature in root.findall('.//feature[@name="plagiarism"]'):
                    source_ref = feature.get('source_reference')
                    
                    if source_ref:
                        # Quitar extensión .txt si existe
                        source_id = source_ref
                        if source_id.endswith('.txt'):
                            source_id = source_id[:-4]
                        
                        # Guardar caso de plagio con toda la información disponible
                        case = {
                            'suspicious_id': suspicious_id,
                            'source_id': source_id,
                            'suspicious_offset': int(feature.get('this_offset')),
                            'suspicious_length': int(feature.get('this_length')),
                            'source_offset': int(feature.get('source_offset')),
                            'source_length': int(feature.get('source_length')),
                            'type': feature.get('type', 'unknown'),
                            'obfuscation': feature.get('obfuscation', 'unknown')
                        }
                        
                        plagiarism_cases.append(case)
                        
                        # Añadir a documentos necesarios
                        source_docs_needed.add(source_id)
                        suspicious_docs_needed.add(suspicious_id)
                    
            except Exception as e:
                print(f"Error al procesar {file_path}: {str(e)}")
    
    print(f"Encontrados {len(plagiarism_cases)} casos de plagio")
    print(f"Documentos fuente necesarios: {len(source_docs_needed)}")
    print(f"Documentos sospechosos necesarios: {len(suspicious_docs_needed)}")
    
    if not plagiarism_cases:
        print("Error: No se encontraron casos de plagio. No se puede continuar.")
        return None
    
    # PASO 2: Cargar los documentos necesarios
    print("Cargando documentos fuente...")
    source_docs = {}
    source_path = os.path.join(corpus_path, "external-detection-corpus", "source-document")
    
    for part_dir in sorted(os.listdir(source_path)):
        if not part_dir.startswith("part"):
            continue
            
        part_path = os.path.join(source_path, part_dir)
        
        for filename in os.listdir(part_path):
            if not filename.endswith(".txt"):
                continue
                
            doc_id = filename[:-4]  # Quitar extensión .txt
            
            if doc_id in source_docs_needed:
                file_path = os.path.join(part_path, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    source_docs[doc_id] = content
                except Exception as e:
                    print(f"Error leyendo {file_path}: {e}")
    
    print("Cargando documentos sospechosos...")
    suspicious_docs = {}
    
    for part_dir in sorted(os.listdir(suspicious_path)):
        if not part_dir.startswith("part"):
            continue
            
        part_path = os.path.join(suspicious_path, part_dir)
        
        for filename in os.listdir(part_path):
            if not filename.endswith(".txt"):
                continue
                
            doc_id = filename[:-4]  # Quitar extensión .txt
            
            if doc_id in suspicious_docs_needed:
                file_path = os.path.join(part_path, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    suspicious_docs[doc_id] = content
                except Exception as e:
                    print(f"Error leyendo {file_path}: {e}")
    
    print(f"Documentos fuente cargados: {len(source_docs)}/{len(source_docs_needed)}")
    print(f"Documentos sospechosos cargados: {len(suspicious_docs)}/{len(suspicious_docs_needed)}")
    
    # PASO 3: Filtrar casos donde tenemos ambos documentos
    valid_cases = []
    for case in plagiarism_cases:
        if case['source_id'] in source_docs and case['suspicious_id'] in suspicious_docs:
            valid_cases.append(case)
    
    print(f"Casos válidos con ambos documentos disponibles: {len(valid_cases)}")
    
    if not valid_cases:
        print("Error: No hay casos válidos con ambos documentos disponibles.")
        return None
    
    # PASO 4: Estratificar casos por tipo de obfuscación para asegurar representatividad
    cases_by_obfuscation = {}
    for case in valid_cases:
        obfuscation = case.get('obfuscation', 'unknown')
        if obfuscation not in cases_by_obfuscation:
            cases_by_obfuscation[obfuscation] = []
        cases_by_obfuscation[obfuscation].append(case)
    
    print("\nDistribución de casos por tipo de obfuscación:")
    for obfuscation, cases in cases_by_obfuscation.items():
        print(f"  - {obfuscation}: {len(cases)} casos ({len(cases)/len(valid_cases)*100:.1f}%)")
    
    # PASO 5: Determinar cuántos ejemplos usar
    # Si max_examples es 0, usar todos los disponibles
    if max_examples == 0:
        # Usar todos los casos disponibles
        total_examples = len(valid_cases)
        print(f"\n¡Usando todos los casos disponibles! Total: {total_examples}")
        # Duplicar el número para crear igual cantidad de negativos
        max_examples = total_examples * 2  
    else:
        total_examples = min(max_examples // 2, len(valid_cases))
        print(f"\nUsando un máximo de {total_examples} ejemplos positivos (de {len(valid_cases)} disponibles)")
        
    # PASO 6: Determinar cuántos ejemplos usar de cada tipo de obfuscación
    examples_per_obfuscation = {}
    
    if max_examples == 0:
        # Usar todos los casos disponibles de cada tipo
        for obfuscation, cases in cases_by_obfuscation.items():
            examples_per_obfuscation[obfuscation] = len(cases)
    else:
        # Distribuir proporcionalmente
        for obfuscation, cases in cases_by_obfuscation.items():
            proportion = len(cases) / len(valid_cases)
            examples_per_obfuscation[obfuscation] = max(5, int(total_examples * proportion))
            
        # Ajustar si la suma no coincide con total_examples
        adjustment_needed = total_examples - sum(examples_per_obfuscation.values())
        
        if adjustment_needed != 0:
            # Distribuir el ajuste proporcionalmente
            for obfuscation in sorted(examples_per_obfuscation.keys(), 
                                    key=lambda k: examples_per_obfuscation[k], 
                                    reverse=(adjustment_needed > 0)):
                if adjustment_needed == 0:
                    break
                    
                if adjustment_needed > 0:
                    examples_per_obfuscation[obfuscation] += 1
                    adjustment_needed -= 1
                else:
                    if examples_per_obfuscation[obfuscation] > 5:  # Mantener al menos 5
                        examples_per_obfuscation[obfuscation] -= 1
                        adjustment_needed += 1
    
    print("\nEjemplos positivos a seleccionar por tipo de obfuscación:")
    for obfuscation, count in examples_per_obfuscation.items():
        print(f"  - {obfuscation}: {count} ejemplos")
    
    # PASO 7: Generar ejemplos positivos y negativos
    examples = []
    
    # Generar ejemplos positivos
    num_positives_target = sum(examples_per_obfuscation.values())
    print(f"\nGenerando {num_positives_target} ejemplos positivos...")
    
    for obfuscation, target_count in examples_per_obfuscation.items():
        cases = cases_by_obfuscation[obfuscation]
        # Seleccionar aleatoriamente casos (en lugar de en orden secuencial)
        selected_cases = random.sample(cases, min(len(cases), target_count))
        
        print(f"Procesando {len(selected_cases)} casos de obfuscación '{obfuscation}'")
        
        # Si estamos en modo batch, mostrar menos información detallada
        if batch_mode:
            iterator = selected_cases
            progress_msg = f"Procesando obfuscación '{obfuscation}'"
            print(progress_msg)
        else:
            iterator = tqdm(selected_cases)
        
        for case in iterator:
            source_text = source_docs[case['source_id']]
            suspicious_text = suspicious_docs[case['suspicious_id']]
            
            # Extraer fragmentos
            try:
                source_fragment = source_text[case['source_offset']:case['source_offset'] + case['source_length']]
                suspicious_fragment = suspicious_text[case['suspicious_offset']:case['suspicious_offset'] + case['suspicious_length']]
                
                if not source_fragment or not suspicious_fragment:
                    continue
                
                # Verificar longitud mínima (para evitar fragmentos demasiado cortos)
                if len(source_fragment.split()) < 10 or len(suspicious_fragment.split()) < 10:
                    continue
                
                # Preprocesar texto
                source_processed = processor.preprocess_text(source_fragment)['processed']
                suspicious_processed = processor.preprocess_text(suspicious_fragment)['processed']
                
                # Extraer características
                features = extractor.extract_features_for_fragment_pair(source_processed, suspicious_processed, update_vocab=True)
                
                example = {
                    'source_doc': case['source_id'],
                    'suspicious_doc': case['suspicious_id'],
                    'source_offset': case['source_offset'],
                    'suspicious_offset': case['suspicious_offset'],
                    'source_fragment': source_fragment[:500],
                    'suspicious_fragment': suspicious_fragment[:500],
                    'is_plagiarism': 1,
                    'obfuscation': case.get('obfuscation', 'unknown'),
                    'svm_features': features['svm_features'],
                    'source_sequence': features['cnn_input']['source_sequence'],
                    'suspicious_sequence': features['cnn_input']['suspicious_sequence']
                }
                
                examples.append(example)
            except Exception as e:
                if not batch_mode:
                    print(f"Error procesando ejemplo positivo: {str(e)}")
    
    num_positives = len(examples)
    print(f"Ejemplos positivos generados: {num_positives}")
    
    # Liberar memoria de documentos que ya no necesitamos
    doc_ids_needed = set()
    for example in examples:
        doc_ids_needed.add(example['source_doc'])
        doc_ids_needed.add(example['suspicious_doc'])
    
    # Filtrar documentos para mantener solo los necesarios
    source_docs = {doc_id: content for doc_id, content in source_docs.items() if doc_id in doc_ids_needed}
    suspicious_docs = {doc_id: content for doc_id, content in suspicious_docs.items() if doc_id in doc_ids_needed}
    
    # Forzar recolección de basura para liberar memoria
    gc.collect()
    
    # PASO 8: Generar ejemplos negativos
    # Intentar generar tantos ejemplos negativos como positivos
    num_negatives_target = num_positives
    print(f"\nGenerando {num_negatives_target} ejemplos negativos...")
    
    # Categorías de similitud para asegurar variabilidad en los ejemplos negativos
    similarity_ranges = {
        'muy_bajo': (0.0, 0.05),  # Casos muy diferentes
        'bajo': (0.05, 0.15),     # Baja similitud
        'medio': (0.15, 0.25),    # Similitud media
        'alto': (0.25, 0.35)      # Alta similitud (casos difíciles pero que no son plagio)
    }
    
    # Cuántos ejemplos generar de cada categoría de similitud
    examples_per_range = {name: num_negatives_target // len(similarity_ranges) for name in similarity_ranges.keys()}
    
    # Ajustar para que sumen exactamente num_negatives_target
    adjustment = num_negatives_target - sum(examples_per_range.values())
    for name in list(examples_per_range.keys())[:adjustment]:
        examples_per_range[name] += 1
    
    # Obtener listas de IDs válidos
    source_ids = list(source_docs.keys())
    suspicious_ids = list(suspicious_docs.keys())
    
    # Clasificar documentos por tema para ejemplos negativos más desafiantes
    source_docs_by_topic = {}
    for s_id in source_ids:
        # Usar los primeros 3 caracteres del ID como "tema"
        topic = s_id[:3]
        if topic not in source_docs_by_topic:
            source_docs_by_topic[topic] = []
        source_docs_by_topic[topic].append(s_id)
    
    # Crear un mapping similar para documentos sospechosos
    suspicious_docs_by_topic = {}
    for s_id in suspicious_ids:
        topic = s_id[:3]
        if topic not in suspicious_docs_by_topic:
            suspicious_docs_by_topic[topic] = []
        suspicious_docs_by_topic[topic].append(s_id)
    
    print(f"\nDocumentos fuente agrupados por {len(source_docs_by_topic)} temas")
    print(f"Documentos sospechosos agrupados por {len(suspicious_docs_by_topic)} temas")
    
    # Distribuir ejemplos negativos proporcionalmente por temas
    print("\nDistribución objetivo de ejemplos negativos por rango de similitud:")
    for name, count in examples_per_range.items():
        min_sim, max_sim = similarity_ranges[name]
        print(f"  - {name} ({min_sim:.2f}-{max_sim:.2f}): {count} ejemplos")
    
    # Contador por categoría
    generated_per_range = {name: 0 for name in similarity_ranges.keys()}
    total_attempts = 0
    max_attempts = num_negatives_target * 20  # Límite para evitar bucles infinitos
    
    # Generar ejemplos negativos
    with tqdm(total=num_negatives_target) as pbar:
        while sum(generated_per_range.values()) < num_negatives_target and total_attempts < max_attempts:
            total_attempts += 1
            
            try:
                # 70% de probabilidad de elegir documentos del mismo "tema" (más desafiante)
                if random.random() < 0.7:
                    common_topics = list(set(source_docs_by_topic.keys()) & set(suspicious_docs_by_topic.keys()))
                    if common_topics:
                        topic = random.choice(common_topics)
                        source_id = random.choice(source_docs_by_topic[topic])
                        suspicious_id = random.choice(suspicious_docs_by_topic[topic])
                    else:
                        source_id = random.choice(source_ids)
                        suspicious_id = random.choice(suspicious_ids)
                else:
                    source_id = random.choice(source_ids)
                    suspicious_id = random.choice(suspicious_ids)
            except Exception as e:
                print(f"Error seleccionando documentos: {e}")
                source_id = random.choice(source_ids)
                suspicious_id = random.choice(suspicious_ids)
                
            # Asegurarse de que los documentos existen
            if source_id not in source_docs or suspicious_id not in suspicious_docs:
                continue
                
            source_text = source_docs[source_id]
            suspicious_text = suspicious_docs[suspicious_id]
            
            # Verificar longitud
            if len(source_text) < 500 or len(suspicious_text) < 500:
                continue
            
            # Seleccionar fragmentos aleatorios
            source_start = random.randint(0, max(0, len(source_text) - 500))
            suspicious_start = random.randint(0, max(0, len(suspicious_text) - 500))
            
            source_length = min(500, len(source_text) - source_start)
            suspicious_length = min(500, len(suspicious_text) - suspicious_start)
            
            source_fragment = source_text[source_start:source_start + source_length]
            suspicious_fragment = suspicious_text[suspicious_start:suspicious_start + suspicious_length]
            
            # Verificar longitud mínima
            if len(source_fragment.split()) < 10 or len(suspicious_fragment.split()) < 10:
                continue
            
            # Calcular similitud
            similarity = extractor._jaccard_similarity(source_fragment, suspicious_fragment)
            
            # Determinar categoría
            selected_range = None
            for name, (min_sim, max_sim) in similarity_ranges.items():
                if min_sim <= similarity < max_sim and generated_per_range[name] < examples_per_range[name]:
                    selected_range = name
                    break
            
            # Si no encaja en ninguna categoría, continuar
            if not selected_range:
                continue
            
            # Extraer características
            try:
                # Preprocesar
                source_processed = processor.preprocess_text(source_fragment)['processed']
                suspicious_processed = processor.preprocess_text(suspicious_fragment)['processed']
                
                # Extraer características
                features = extractor.extract_features_for_fragment_pair(source_processed, suspicious_processed, update_vocab=True)
                
                example = {
                    'source_doc': source_id,
                    'suspicious_doc': suspicious_id,
                    'source_offset': source_start,
                    'suspicious_offset': suspicious_start,
                    'source_fragment': source_fragment[:500],
                    'suspicious_fragment': suspicious_fragment[:500],
                    'is_plagiarism': 0,
                    'obfuscation': 'none',
                    'similarity': similarity,
                    'similarity_range': selected_range,
                    'svm_features': features['svm_features'],
                    'source_sequence': features['cnn_input']['source_sequence'],
                    'suspicious_sequence': features['cnn_input']['suspicious_sequence']
                }
                
                examples.append(example)
                generated_per_range[selected_range] += 1
                
                # Actualizar barra de progreso
                negative_count = sum(generated_per_range.values())
                pbar.update(negative_count - pbar.n)
                
                # Mostrar progreso detallado cada 500 ejemplos o 10% del total
                update_interval = max(500, num_negatives_target // 10)
                if negative_count % update_interval == 0 and negative_count > 0 and not batch_mode:
                    print(f"\nDistribución actual de ejemplos negativos:")
                    for name, count in generated_per_range.items():
                        min_sim, max_sim = similarity_ranges[name]
                        print(f"  - {name} ({min_sim:.2f}-{max_sim:.2f}): {count}/{examples_per_range[name]} ejemplos")
                
            except Exception as e:
                if not batch_mode:
                    print(f"Error procesando ejemplo negativo: {str(e)}")
    
    # Informe sobre ejemplos negativos generados
    num_negatives = sum(generated_per_range.values())
    print("\nEjemplos negativos generados por rango de similitud:")
    for name, count in generated_per_range.items():
        min_sim, max_sim = similarity_ranges[name]
        print(f"  - {name} ({min_sim:.2f}-{max_sim:.2f}): {count}/{examples_per_range[name]} ejemplos ({count/num_negatives*100:.1f}%)")
    
    print(f"\nTotal: {num_negatives}/{num_negatives_target} ejemplos negativos")
    
    # PASO 9: Comprobar el total de ejemplos
    total_examples = len(examples)
    print(f"\nTotal de ejemplos generados: {total_examples} ({num_positives} positivos, {num_negatives} negativos)")
    
    if total_examples == 0:
        print("Error: No se generaron ejemplos. No se puede continuar.")
        return None
    
    # PASO 10: Preparar y guardar dataset
    print("\nProcesando ejemplos y preparando dataset...")
    
    # Convertir a DataFrame para facilitar manejo
    examples_df = pd.DataFrame([
        {
            'source_doc': ex['source_doc'],
            'suspicious_doc': ex['suspicious_doc'],
            'source_offset': ex['source_offset'],
            'suspicious_offset': ex['suspicious_offset'],
            'source_fragment': ex['source_fragment'],
            'suspicious_fragment': ex['suspicious_fragment'],
            'is_plagiarism': ex['is_plagiarism'],
            'obfuscation': ex['obfuscation'],
            'similarity': ex.get('similarity', 1.0),  # Para ejemplos positivos, asumimos similitud 1.0
            'similarity_range': ex.get('similarity_range', 'positive')  # Para ejemplos positivos
        } for ex in examples
    ])
    
    # Extraer matrices de características
    print("Extrayendo matrices de características...")
    X_svm = np.vstack([ex['svm_features'] for ex in examples])
    X_cnn_source = np.vstack([ex['source_sequence'].reshape(1, -1) for ex in examples])
    X_cnn_suspicious = np.vstack([ex['suspicious_sequence'].reshape(1, -1) for ex in examples])
    y = np.array([ex['is_plagiarism'] for ex in examples])
    
    # Dividir en entrenamiento y prueba, estratificando por etiqueta y tipo de obfuscación
    print("Dividiendo en conjuntos de entrenamiento y prueba...")
    indices = np.arange(len(examples))
    
    # Crear una estratificación combinada: etiqueta + obfuscación
    stratify_labels = [f"{ex['is_plagiarism']}_{ex['obfuscation']}" for ex in examples]
    
    # Usar train_test_split con estratificación
    indices_train, indices_test = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=stratify_labels)
    
    # Verificar distribución en train y test
    train_distribution = collections.Counter([examples[i]['obfuscation'] for i in indices_train])
    test_distribution = collections.Counter([examples[i]['obfuscation'] for i in indices_test])
    
    print("\nDistribución de tipos de obfuscación en conjunto de entrenamiento:")
    for obfuscation, count in train_distribution.items():
        print(f"  - {obfuscation}: {count} ejemplos ({count/len(indices_train)*100:.1f}%)")
    
    print("\nDistribución de tipos de obfuscación en conjunto de prueba:")
    for obfuscation, count in test_distribution.items():
        print(f"  - {obfuscation}: {count} ejemplos ({count/len(indices_test)*100:.1f}%)")
    
    # Preparar datos
    dataset = {
        'train': {
            'X_svm': X_svm[indices_train],
            'X_cnn_source': X_cnn_source[indices_train],
            'X_cnn_suspicious': X_cnn_suspicious[indices_train],
            'y': y[indices_train],
            'metadata': examples_df.iloc[indices_train]
        },
        'test': {
            'X_svm': X_svm[indices_test],
            'X_cnn_source': X_cnn_source[indices_test],
            'X_cnn_suspicious': X_cnn_suspicious[indices_test],
            'y': y[indices_test],
            'metadata': examples_df.iloc[indices_test]
        },
        'vocab_size': extractor.vocab_size,
        'max_sequence_length': extractor.max_sequence_length
    }
    
    # Guardar dataset
    print("Guardando dataset en disco...")
    os.makedirs('data/processed', exist_ok=True)
    
    np.save('data/processed/X_svm_train.npy', dataset['train']['X_svm'])
    np.save('data/processed/X_cnn_source_train.npy', dataset['train']['X_cnn_source'])
    np.save('data/processed/X_cnn_suspicious_train.npy', dataset['train']['X_cnn_suspicious'])
    np.save('data/processed/y_train.npy', dataset['train']['y'])
    
    np.save('data/processed/X_svm_test.npy', dataset['test']['X_svm'])
    np.save('data/processed/X_cnn_source_test.npy', dataset['test']['X_cnn_source'])
    np.save('data/processed/X_cnn_suspicious_test.npy', dataset['test']['X_cnn_suspicious'])
    np.save('data/processed/y_test.npy', dataset['test']['y'])
    
    dataset['train']['metadata'].to_csv('data/processed/metadata_train.csv', index=False)
    dataset['test']['metadata'].to_csv('data/processed/metadata_test.csv', index=False)
    
    # Guardar información detallada del dataset
    with open('data/processed/dataset_info.txt', 'w') as f:
        f.write(f"Total examples: {len(examples)}\n")
        f.write(f"Positive examples: {num_positives}\n")
        f.write(f"Negative examples: {num_negatives}\n")
        f.write(f"SVM features dimension: {dataset['vocab_size']}\n")
        f.write(f"CNN sequence length: {dataset['max_sequence_length']}\n")
        f.write(f"Language: {language}\n")
        f.write(f"Pretrained embeddings: {use_pretrained_embeddings}\n")
        f.write(f"Embedding model: {embedding_model if use_pretrained_embeddings else 'None'}\n\n")
        
        # Distribución detallada por tipo de obfuscación
        f.write("=== Distribution by obfuscation type ===\n")
        obfuscation_counts = collections.Counter([ex['obfuscation'] for ex in examples])
        for obfuscation, count in obfuscation_counts.items():
            f.write(f"{obfuscation}: {count} examples ({count/len(examples)*100:.1f}%)\n")
        
        # Distribución de ejemplos negativos por rango de similitud
        f.write("\n=== Distribution of negative examples by similarity range ===\n")
        negative_examples = [ex for ex in examples if ex['is_plagiarism'] == 0]
        for name in similarity_ranges.keys():
            count = sum(1 for ex in negative_examples if ex.get('similarity_range') == name)
            f.write(f"{name}: {count} examples ({count/len(negative_examples)*100:.1f}%)\n")
    
    # Guardar vocabulario para uso posterior
    extractor.save_vocabulary('data/processed/vocabulary.pkl')
    
    # Generar y guardar matriz de embeddings si se han utilizado
    if use_pretrained_embeddings:
        print("Generando matriz de embeddings...")
        embedding_matrix = extractor.create_embedding_matrix()
        np.save('data/processed/embedding_matrix.npy', embedding_matrix)
    
    # Tiempo total de procesamiento
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTiempo total de procesamiento: {total_time:.2f} segundos")
    print("Dataset creado y guardado con éxito.")
    
    return dataset

if __name__ == "__main__":
    # Añadir argumentos para permitir diferentes configuraciones
    parser = argparse.ArgumentParser(description='Crear dataset para detección de plagio')
    parser.add_argument('--corpus', type=str, default=os.path.join('data', 'pan-plagiarism-corpus-2011'),
                        help='Ruta al corpus PAN')
    parser.add_argument('--max_examples', type=int, default=0,
                        help='Número máximo de ejemplos a generar (0 para usar todos los disponibles)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proporción del dataset para prueba')
    parser.add_argument('--language', type=str, default='english', choices=['english', 'spanish', 'german'],
                        help='Idioma del corpus')
    parser.add_argument('--use_pretrained', type=bool, default=True,
                        help='Usar embeddings preentrenados')
    parser.add_argument('--embedding_model', type=str, default='glove-wiki-gigaword-300',
                        help='Modelo de embeddings a utilizar')
    parser.add_argument('--batch', action='store_true',
                        help='Procesar en modo batch con reportes de progreso reducidos')
    
    args = parser.parse_args()
    
    print(f"Configuración: max_examples={args.max_examples} (0 = usar todos los disponibles)")
    
    dataset = create_dataset(
        corpus_path=args.corpus,
        max_examples=args.max_examples,
        test_size=args.test_size,
        language=args.language,
        use_pretrained_embeddings=args.use_pretrained,
        embedding_model=args.embedding_model,
        batch_mode=args.batch
    )
    
    if dataset:
        print("Dataset creado y guardado en data/processed/")
    else:
        print("No se pudo crear el dataset")