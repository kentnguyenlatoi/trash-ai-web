import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
layers = tf.keras.layers


tf.keras.backend.clear_session()
DATA_DIR = "Dataset_flat"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR + "/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR + "/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)


# Data augmentation
data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
])

# Base model
base = keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    weights="imagenet"
)
base.trainable = False  # giai đoạn 1: freeze

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_aug(inputs)
x = keras.applications.efficientnet.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
]

print("== Train stage 1 ==")
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)

# Fine-tune một chút (giai đoạn 2)
print("== Fine-tune stage 2 ==")
base.trainable = True
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=callbacks)

# Save final
model.save("trash_classifier.keras")
with open("labels.txt", "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print("✅ Saved: trash_classifier.keras + labels.txt")
