import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image
import seaborn as sns

# Configurar GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✅ GPU configurada correctamente")
else:
    print("⚠️  No se encontró GPU, usando CPU")

# ⚠️ PARÁMETROS REDUCIDOS PARA EVITAR OUT OF MEMORY
IMG_SIZE = (96, 96)  # Reducir tamaño de imagen
BATCH_SIZE = 16      # Reducir batch size
EPOCHS = 50
EMBEDDING_DIM = 128  # Reducir dimensión de embeddings

DATA_PATH = "/kaggle/input/dataset-images/datos_procesados_01"

def load_dataset(data_path, max_images_per_class=15):
    """Carga el dataset con límite de imágenes por clase"""
    images = []
    labels = []
    label_names = []
    
    person_folders = sorted([f for f in os.listdir(data_path) 
                           if os.path.isdir(os.path.join(data_path, f))])
    
    print(f"Encontradas {len(person_folders)} personas")
    
    for label, person_folder in enumerate(person_folders):
        folder_path = os.path.join(data_path, person_folder)
        label_names.append(person_folder)
        
        # Limitar imágenes por clase
        img_files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        img_files = img_files[:max_images_per_class]
        
        for img_file in img_files:
            img_path = os.path.join(folder_path, img_file)
            
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(IMG_SIZE)
                img_array = np.array(img) / 255.0
                
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error cargando {img_path}: {e}")
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Dataset cargado: {len(images)} imágenes, {len(label_names)} clases")
    return images, labels, label_names

# Cargar dataset con límites
print("📁 Cargando dataset...")
X, y, label_names = load_dataset(DATA_PATH, max_images_per_class=15)
NUM_CLASSES = len(label_names)
print(f"Número de clases: {NUM_CLASSES}")

# Dividir dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Entrenamiento: {X_train.shape[0]} imágenes")
print(f"Validación: {X_val.shape[0]} imágenes")

# Liberar memoria eliminando X, y originales
del X, y

def create_lightweight_model(input_shape, num_classes, embedding_dim=128):
    """Modelo más ligero para ahorrar memoria"""
    
    inputs = keras.Input(shape=input_shape)
    
    # Bloque 1 - Reducido
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    # Bloque 2
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    # Bloque 3
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    # Capas fully connected reducidas
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Embeddings
    embeddings = layers.Dense(embedding_dim, name='embedding')(x)
    embeddings = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), 
                             name='normalized_embedding')(embeddings)
    
    # Clasificación
    classification = layers.Dense(num_classes, activation='softmax', 
                                name='classification')(x)
    
    model = keras.Model(inputs=inputs, 
                       outputs=[embeddings, classification])
    
    return model

# Crear modelo ligero
print("🔄 Creando modelo...")
model = create_lightweight_model(
    input_shape=(*IMG_SIZE, 3), 
    num_classes=NUM_CLASSES, 
    embedding_dim=EMBEDDING_DIM
)

model.summary()

# Compilar modelo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'normalized_embedding': 'mse',  # Usar MSE temporalmente en lugar de triplet loss
        'classification': 'sparse_categorical_crossentropy'
    },
    loss_weights={
        'normalized_embedding': 0.4,
        'classification': 0.6
    },
    metrics={
        'classification': ['accuracy']
    }
)

# Callbacks simplificados
callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_classification_accuracy',
        factor=0.5,
        patience=8,
        min_lr=1e-7
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_classification_accuracy',
        patience=15,
        restore_best_weights=True
    )
]

# Entrenar en lotes más pequeños si es necesario
print("🎯 Iniciando entrenamiento...")

# Si aún hay problemas de memoria, usar fit en lotes más pequeños
try:
    history = model.fit(
        X_train,
        {
            'normalized_embedding': y_train,
            'classification': y_train
        },
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(
            X_val,
            {
                'normalized_embedding': y_val,
                'classification': y_val
            }
        ),
        callbacks=callbacks,
        verbose=1
    )
    
    # Guardar modelo
    model.save('face_recognition_light.h5')
    print("✅ Modelo guardado como 'face_recognition_light.h5'")
    
    # Generar gráficas
    def plot_training_history(history):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(history.history['loss'], label='Entrenamiento')
        axes[0, 0].plot(history.history['val_loss'], label='Validación')
        axes[0, 0].set_title('Pérdida Total')
        axes[0, 0].legend()
        
        axes[0, 1].plot(history.history['classification_accuracy'], label='Entrenamiento')
        axes[0, 1].plot(history.history['val_classification_accuracy'], label='Validación')
        axes[0, 1].set_title('Precisión de Clasificación')
        axes[0, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history_light.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    plot_training_history(history)
    
except Exception as e:
    print(f"❌ Error durante el entrenamiento: {e}")
    print("💡 Intentando con batch size más pequeño...")
    
    # Último intento con batch size muy pequeño
    history = model.fit(
        X_train,
        {
            'normalized_embedding': y_train,
            'classification': y_train
        },
        batch_size=8,  # Batch size mínimo
        epochs=EPOCHS,
        validation_data=(
            X_val,
            {
                'normalized_embedding': y_val,
                'classification': y_val
            }
        ),
        callbacks=callbacks,
        verbose=1
    )