from keras.applications import ResNet50

model = ResNet50(include_top=True, weights="imagenet")
print(model.summary())