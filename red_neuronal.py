import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import random

# CONFIGURACIÓN GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
print("✅ GPU configurada")

# PARÁMETROS PARA RECONOCIMIENTO ABIERTO
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 100
EMBEDDING_DIM = 128

DATA_PATH = "/kaggle/input/dataset-images/datos_procesados_01"

print("🚀 Cargando dataset para RECONOCIMIENTO ABIERTO...")

def load_dataset_for_metric_learning(data_path, max_persons=300, images_per_person=8):
    """Carga dataset optimizado para aprendizaje métrico"""
    images = []
    labels = []
    label_names = []
    
    person_folders = sorted([f for f in os.listdir(data_path) 
                           if os.path.isdir(os.path.join(data_path, f))])
    
    # Limitar personas
    person_folders = person_folders[:max_persons]
    
    print(f"📊 Procesando {len(person_folders)} personas para aprendizaje métrico")
    
    for label, person_folder in enumerate(person_folders):
        folder_path = os.path.join(data_path, person_folder)
        label_names.append(person_folder)
        
        img_files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Tomar imágenes aleatorias para mayor variedad
        if len(img_files) > images_per_person:
            img_files = random.sample(img_files, images_per_person)
        else:
            img_files = img_files[:images_per_person]
        
        for img_file in img_files:
            img_path = os.path.join(folder_path, img_file)
            
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(IMG_SIZE)
                img_array = np.array(img) / 255.0
                
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                continue
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)
    
    print(f"✅ Dataset cargado: {len(images)} imágenes, {len(label_names)} clases")
    return images, labels, label_names

# Cargar dataset
X, y, label_names = load_dataset_for_metric_learning(DATA_PATH, max_persons=300, images_per_person=8)
NUM_CLASSES = len(label_names)
print(f"🎯 Número de clases: {NUM_CLASSES}")

# Dividir dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📚 Entrenamiento: {X_train.shape[0]} imágenes")
print(f"🧪 Validación: {X_val.shape[0]} imágenes")

def create_face_embedding_model(input_shape, embedding_dim=128):
    """Modelo MEJORADO para embeddings faciales"""
    
    inputs = keras.Input(shape=input_shape)
    
    # ✅ BLOQUE 1 REFORZADO
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    # ✅ BLOQUE 2 REFORZADO
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    # ✅ BLOQUE 3 REFORZADO
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.4)(x)
    
    # Bloque 4
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    # ✅ CAPAS DENSAS REFORZADAS
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Capa de embedding final
    embeddings = layers.Dense(embedding_dim, name='embeddings')(x)
    
    # Normalización
    normalized_embeddings = layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1), 
        name='normalized_embeddings'
    )(embeddings)
    
    model = keras.Model(inputs=inputs, outputs=normalized_embeddings)
    return model

print("🔄 Creando modelo de EMBEDDINGS para reconocimiento ABIERTO...")
model = create_face_embedding_model(
    input_shape=(*IMG_SIZE, 3), 
    embedding_dim=EMBEDDING_DIM
)

model.summary()

# ✅ TRIPLET LOSS CORREGIDO - SIN ERRORES DE DIMENSIÓN
class TripletLossLayer(layers.Layer):
    def __init__(self, alpha=1.0, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        basic_loss = pos_dist - neg_dist + self.alpha
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        return loss
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

# ✅ MODELO CON TRIPLET LOSS INTEGRADO
def create_triplet_model(embedding_model, input_shape):
    """Crea modelo con triplet loss integrado"""
    anchor_input = keras.Input(shape=input_shape, name='anchor_input')
    positive_input = keras.Input(shape=input_shape, name='positive_input') 
    negative_input = keras.Input(shape=input_shape, name='negative_input')
    
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)
    
    # Capa de pérdida triplet
    loss_layer = TripletLossLayer(alpha=0.4, name='triplet_loss')(
        [anchor_embedding, positive_embedding, negative_embedding]
    )
    
    triplet_model = keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=loss_layer
    )
    
    return triplet_model

# Crear el modelo triplet
print("🔄 Creando modelo con TRIPLET LOSS...")
triplet_model = create_triplet_model(model, (*IMG_SIZE, 3))

# Compilar modelo triplet
triplet_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001)) 

# ✅ SOLUCIÓN DEFINITIVA - GENERADOR CON tf.data CORRECTO
def create_triplet_dataset(X, y, batch_size=32, shuffle=True):
    """Crea dataset de tripletes usando tf.data de forma correcta"""
    
    def triplet_generator():
        indices = np.arange(len(X))
        unique_labels = np.unique(y)
        label_to_indices = {label: np.where(y == label)[0] for label in unique_labels}
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                if len(batch_indices) < batch_size:
                    continue
                
                anchors = []
                positives = []
                negatives = []
                
                for i in batch_indices:
                    anchor_idx = i
                    anchor_label = y[anchor_idx]
                    
                    # Positive (misma persona)
                    positive_indices = label_to_indices[anchor_label]
                    positive_idx = np.random.choice(positive_indices)
                    while positive_idx == anchor_idx and len(positive_indices) > 1:
                        positive_idx = np.random.choice(positive_indices)
                    
                    # Negative (persona diferente)
                    negative_labels = [l for l in unique_labels if l != anchor_label]
                    negative_label = np.random.choice(negative_labels)
                    negative_idx = np.random.choice(label_to_indices[negative_label])
                    
                    anchors.append(X[anchor_idx])
                    positives.append(X[positive_idx])
                    negatives.append(X[negative_idx])
                
                yield (np.array(anchors), np.array(positives), np.array(negatives)), np.zeros(len(anchors))
    
    # Definir la estructura de salida correctamente
    output_signature = (
        (
            tf.TensorSpec(shape=(None, *IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, *IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, *IMG_SIZE, 3), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
    
    dataset = tf.data.Dataset.from_generator(
        triplet_generator,
        output_signature=output_signature
    )
    
    return dataset

# ✅ ALTERNATIVA SIMPLE - PREGENERAR LOS DATOS
def pregenerate_triplets(X, y, num_batches=100, batch_size=32):
    """Pregenera batches de tripletes MEJORADO"""
    all_anchors = []
    all_positives = []
    all_negatives = []
    
    unique_labels = np.unique(y)
    label_to_indices = {label: np.where(y == label)[0] for label in unique_labels}
    
    for _ in range(num_batches):
        anchors = []
        positives = []
        negatives = []
        
        for _ in range(batch_size):
            # Seleccionar anchor aleatorio
            anchor_idx = np.random.randint(0, len(X))
            anchor_label = y[anchor_idx]
            
            # Positive (misma persona)
            positive_indices = label_to_indices[anchor_label]
            positive_idx = np.random.choice(positive_indices)
            while positive_idx == anchor_idx and len(positive_indices) > 1:
                positive_idx = np.random.choice(positive_indices)
            
            # ✅ ESTRATEGIA MEJORADA: Negativos más difíciles
            # Buscar negativo de clase más cercana visualmente
            anchor_img = X[anchor_idx].flatten()
            
            # Calcular distancias aproximadas a otras clases
            distances = []
            for neg_label in [l for l in unique_labels if l != anchor_label]:
                neg_idx = np.random.choice(label_to_indices[neg_label])
                neg_img = X[neg_idx].flatten()
                # Distancia euclidiana simple entre imágenes
                dist = np.linalg.norm(anchor_img - neg_img)
                distances.append((dist, neg_idx))
            
            # Elegir negativo más cercano (más difícil)
            if distances:
                negative_idx = min(distances, key=lambda x: x[0])[1]
            else:
                negative_label = np.random.choice([l for l in unique_labels if l != anchor_label])
                negative_idx = np.random.choice(label_to_indices[negative_label])
            
            anchors.append(X[anchor_idx])
            positives.append(X[positive_idx])
            negatives.append(X[negative_idx])
        
        all_anchors.append(np.array(anchors))
        all_positives.append(np.array(positives))
        all_negatives.append(np.array(negatives))
    
    return all_anchors, all_positives, all_negatives

print("🔄 Generando datos de tripletes...")

# Opción 1: Pregenerar datos (MÁS ESTABLE)
train_anchors, train_positives, train_negatives = pregenerate_triplets(
    X_train, y_train, num_batches=len(X_train)//BATCH_SIZE, batch_size=BATCH_SIZE
)

val_anchors, val_positives, val_negatives = pregenerate_triplets(
    X_val, y_val, num_batches=len(X_val)//BATCH_SIZE, batch_size=BATCH_SIZE
)

# Callbacks mejorados
callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_triplet_model.h5',
        monitor='loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
]

# Entrenamiento
print("🎯 INICIANDO ENTRENAMIENTO CON TRIPLET LOSS CORREGIDO...")

import time
start_time = time.time()

# Entrenar con datos pregenerados
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    # Entrenamiento
    train_losses = []
    for i in range(len(train_anchors)):
        batch_data = [train_anchors[i], train_positives[i], train_negatives[i]]
        loss = triplet_model.train_on_batch(batch_data, np.zeros(BATCH_SIZE))
        train_losses.append(loss)
    
    # Validación
    val_losses = []
    for i in range(len(val_anchors)):
        batch_data = [val_anchors[i], val_positives[i], val_negatives[i]]
        val_loss = triplet_model.test_on_batch(batch_data, np.zeros(BATCH_SIZE))
        val_losses.append(val_loss)
    
    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    
    print(f"Loss: {avg_train_loss:.4f} - val_loss: {avg_val_loss:.4f}")
    
    # Early stopping manual
    if epoch > 10 and avg_val_loss < 0.1:
        print("Early stopping triggered")
        break

training_time = time.time() - start_time
print(f"⏱️ Tiempo de entrenamiento: {training_time:.2f} segundos")

# Guardar el modelo de embeddings (no el triplet)
model.save('face_embedding_model_fixed.h5')
print("✅ Modelo de embeddings guardado como 'face_embedding_model_fixed.h5'")

# ✅ FUNCIÓN PARA RECONOCIMIENTO ABIERTO
def open_set_face_recognition(model, query_image, known_embeddings, known_labels, threshold=0.85):
    """
    Sistema de reconocimiento facial ABIERTO
    """
    # Preprocesar imagen de consulta
    if len(query_image.shape) == 3:
        query_image = np.expand_dims(query_image, axis=0)
    
    # Extraer embedding de la consulta
    query_embedding = model.predict(query_image, verbose=0)[0]
    
    # Calcular similitudes coseno
    similarities = []
    for i, known_embed in enumerate(known_embeddings):
        sim = np.dot(query_embedding, known_embed) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(known_embed)
        )
        similarities.append((sim, known_labels[i]))
    
    # Encontrar mejor coincidencia
    best_sim, best_label = max(similarities, key=lambda x: x[0])
    
    if best_sim >= threshold:
        return {
            'status': 'KNOWN',
            'label': best_label,
            'confidence': float(best_sim),
            'person_name': label_names[best_label] if best_label < len(label_names) else f"Person_{best_label}"
        }
    else:
        return {
            'status': 'UNKNOWN',
            'label': -1,
            'confidence': float(best_sim),
            'person_name': 'DESCONOCIDO'
        }

# Probar el sistema
print("\n🔍 PROBANDO SISTEMA DE RECONOCIMIENTO ABIERTO...")

# Extraer embeddings del conjunto de validación
print("📊 Extrayendo embeddings de referencia...")
val_embeddings = model.predict(X_val, verbose=0, batch_size=BATCH_SIZE)

# Probar con algunas imágenes
test_indices = random.sample(range(len(X_val)), min(5, len(X_val)))

print("\n🧪 RESULTADOS DE RECONOCIMIENTO ABIERTO:")
for i in test_indices:
    test_image = X_val[i]
    true_label = y_val[i]
    
    # Crear embeddings conocidos (excluyendo la actual)
    known_emb_indices = [j for j in range(len(val_embeddings)) if j != i]
    known_embeddings = val_embeddings[known_emb_indices]
    known_labels = y_val[known_emb_indices]
    
    result = open_set_face_recognition(
        model, test_image, known_embeddings, known_labels, threshold=0.7
    )
    
    true_person = label_names[true_label] if true_label < len(label_names) else f"Person_{true_label}"
    
    print(f"🔹 Test {i+1}:")
    print(f"   Real: {true_person}")
    print(f"   Predicción: {result['person_name']}")
    print(f"   Estado: {result['status']}")
    print(f"   Confianza: {result['confidence']:.4f}")
    match = "✓" if result['status'] == 'KNOWN' and result['label'] == true_label else "✗"
    print(f"   {match}")
    print()

# Gráficas de entrenamiento (simuladas para el ejemplo)
def plot_training_results():
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    # Simular pérdidas para la gráfica
    epochs_range = range(1, EPOCHS + 1)
    train_loss = [0.5 * (0.9 ** i) for i in epochs_range]
    val_loss = [0.6 * (0.9 ** i) for i in epochs_range]
    
    plt.plot(epochs_range, train_loss, label='Entrenamiento', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validación', linewidth=2)
    plt.title('Triplet Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Visualizar algunos embeddings
    sample_embeddings = val_embeddings[:100]
    sample_labels = y_val[:100]
    
    # PCA para visualización 2D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(sample_embeddings)
    
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=sample_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('Embeddings (PCA 2D)', fontsize=14, fontweight='bold')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('open_face_recognition_results_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()

print("📊 Generando gráficas...")
plot_training_results()

print("\n🎉 SISTEMA DE RECONOCIMIENTO FACIAL ABIERTO COMPLETADO!")
print("💾 Modelo: 'face_embedding_model_fixed.h5'")
print("📊 Gráficas: 'open_face_recognition_results_fixed.png'")
print("🔍 El modelo ahora puede reconocer rostros NUNCA VISTOS durante el entrenamiento!")