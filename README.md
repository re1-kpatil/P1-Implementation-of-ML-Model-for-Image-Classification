# P1-Implementation-of-ML-Model-for-Image-Classification
This project is a web-based image classifier using Streamlit and a CNN trained on the Fashion MNIST dataset. Users can upload clothing images, and the app predicts the category (e.g., T-shirt, Sneaker) with confidence scores. Built with Python, TensorFlow, Streamlit, and deployed on Streamlit Cloud.
## Features
- User-Friendly Interface: Built with Streamlit, allowing users to upload images effortlessly.
- Real-Time Predictions: Displays the predicted class and confidence percentage for uploaded images.
- Efficient CNN Model: A lightweight CNN model trained on the Fashion MNIST dataset for fast and accurate predictions.
- Deployment: Hosted on Streamlit Cloud for easy access without local installation.
- Customizable: Designed to be extendable for other datasets or more complex models.
## Tech Stack
- Frontend: Streamlit
- Backend: Python and Streamlit API
- Machine Learning Frameworks: TensorFlow, Keras, scikit-learn
- Deployment: Streamlit Cloud
- Development Tools: Jupyter Notebook, Anaconda
## How It Works
1. Model Training:
   - A CNN model was trained on the Fashion MNIST dataset (grayscale 28x28 images of clothing items).
   - The model achieves high accuracy with a lightweight architecture optimized for machines without a GPU.
2. User Interaction:
   - Users upload a .png image of a clothing item through the Streamlit app.
   - The app preprocesses the image (resizing and normalization) and passes it to the trained model for prediction.
3. Output:
   - The app displays the predicted clothing category (e.g., T-shirt, Sneaker) and the model's confidence in percentage.
## Dataset
### - Fashion MNIST Dataset:
- Contains 60,000 training images and 10,000 testing images of clothing items in 10 categories.
- Categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot.
## Future Enhancements
- Add support for color images and other datasets.
- Implement advanced models like MobileNet (with optimizations for low-resource environments).
- Include additional preprocessing options for uploaded images.




