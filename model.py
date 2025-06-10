#Summary of Model
from tensorflow.keras.models import load_model
model = load_model('./model_vgg16_finetuned.keras')
model.summary()