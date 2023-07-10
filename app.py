import streamlit as st
import cv2
import numpy as np
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import keras

st.title("Brain Tumor Prediction Using CNN")

uploaded_file = st.file_uploader("Upload MRI Image of Brain")
st.write("Uploaded Image")

predict = False

if uploaded_file is not None:
    # Read the image data from the uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Display the original image on the web page
    st.image(img, channels="BGR")
    predict = st.button("Predict")

if predict:
    model = load_model("my_model.h5")
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    imagetestarray = np.zeros((1, 3, 128, 128))
    resized_image = cv2.resize(img, (128, 128))
    sharpened_image = cv2.filter2D(resized_image, -1, kernel)
    st.image(sharpened_image, channels="BGR")
    image_array = np.array(sharpened_image, dtype=np.float32).transpose((2, 0, 1)) / 128
    imagetestarray[0] = image_array
    imagetestarray = imagetestarray.reshape((1, 128, 128, 3))

    # Define a new model to output feature maps at each layer
    layer_outputs = [layer.output for layer in model.layers[:6]]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    # Get the feature maps for the input image
    activations = activation_model.predict(imagetestarray)

    for i in range(len(activations)):
        layer_name = model.layers[i].name
        activation = activations[i]
        fig, axs = plt.subplots(4, 8, figsize=(16, 8))
        fig.suptitle(layer_name, fontsize=16)
        for j in range(32):
            axs[j // 8, j % 8].imshow(activation[0, :, :, j], cmap='gray')
            axs[j // 8, j % 8].axis('off')
            axs[j // 8, j % 8].set_title(f"map {j}")
        st.pyplot(fig)

        # Display the kernel used for the current layer
        if isinstance(model.layers[i], keras.layers.Conv2D):
            kernel = model.layers[i].get_weights()[0]
            fig2, axs2 = plt.subplots(kernel.shape[-1], kernel.shape[-2], figsize=(16, 8))
            fig2.suptitle(layer_name + ' Kernel', fontsize=16)
            for k in range(kernel.shape[-1]):
                for l in range(kernel.shape[-2]):
                    axs2[k, l].imshow(kernel[:, :, l, k], cmap='gray')
                    axs2[k, l].axis('off')
            st.pyplot(fig2)

    # Make a prediction on the input image
    prediction = model.predict(imagetestarray)
    predicted_label = np.argmax(prediction)
    if predicted_label == 0:
        st.header("glioma")
    elif predicted_label == 1:
        st.header("Meningioma")
    elif predicted_label == 2:
        st.subheader("Pritutory Tumor")










