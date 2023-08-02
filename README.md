# Superhero Name Generator with TensorFlow

This project aims to create a superhero name generator using TensorFlow. The goal is to train a neural network model on a dataset containing more than 9000 names of superheroes and supervillains from various comic books, TV shows, and movies. The trained model will then be able to generate new superhero names based on the patterns it learns from the dataset.

## How it Works

The project uses a character-level language model, where the neural network predicts the next character in a sequence given a seed input. The seed input can be a single character or a sequence of characters. The model generates a new character, adds it to the seed input, and uses the extended sequence to predict the next character. This process continues iteratively, creating a sequence of characters that eventually forms a superhero name.

## Architecture
In this project, we use a specific architecture for our superhero name generator. The architecture consists of several layers that process the input data and generate the superhero names. Here's a breakdown of the architecture:

1. **Embedding Layer (Embedding)**:
   - Input Shape: (None, 32)
   - Output Shape: (None, 32, 8)
   - Number of Parameters: 232

2. **Convolutional 1D Layer (Conv1D)**:
   - Input Shape: (None, 32, 8)
   - Output Shape: (None, 32, 64)
   - Number of Parameters: 2,624

3. **Max Pooling 1D Layer (MaxPooling1D)**:
   - Input Shape: (None, 32, 64)
   - Output Shape: (None, 16, 64)

4. **LSTM Layer (LSTM)**:
   - Input Shape: (None, 16, 64)
   - Output Shape: (None, 32)
   - Number of Parameters: 12,416

5. **Dense Layer (Dense)**:
   - Input Shape: (None, 32)
   - Output Shape: (None, 29)
   - Number of Parameters: 957

This architecture takes a sequence of 32 characters as input and uses an embedding layer to convert each character into an 8-dimensional vector representation. The Conv1D layer applies a 1D convolution operation to the embedded sequence, followed by a max pooling operation to reduce the sequence length. The LSTM layer processes the pooled sequence and outputs a 32-dimensional representation. Finally, the Dense layer generates the output, predicting the next character in the sequence from a vocabulary of 29 characters (26 alphabet characters + 3 special characters).
