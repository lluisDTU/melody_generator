# Melody Continuation Generator

Melody Continuation Generator is a deep learning model designed to generate creative continuations of MIDI melodies using LSTMs. The model is trained on a dataset of MIDI files to learn musical patterns and generate new melodies based on a given input melody.

## Getting Started

These instructions will help you set up the project and run it on your local machine.

### Prerequisites

- Python 3.7 or higher
- Install required Python packages:

`pip install -r requirements.txt


### Usage

1. Place the input MIDI melody file (e.g., `Melody.mid`) in the project directory.
2. Run the script `generate_melody.py`:


`python generate_melody.py`

3. The generated melody continuation will be saved as a new MIDI file (e.g., `new_melody.mid`) in the project directory.

## Built With

- [music21](http://web.mit.edu/music21/) - A toolkit for computer-aided musicology
- [Keras](https://keras.io/) - A high-level neural networks API
- [TensorFlow](https://www.tensorflow.org/) - An open-source machine learning framework

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The code in this repository is based on the suggestions and guidance provided by OpenAI's ChatGPT.
