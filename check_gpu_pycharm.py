import tensorflow as tf

print("System Check:")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Found GPU: {tf.config.list_physical_devices('GPU')}")
