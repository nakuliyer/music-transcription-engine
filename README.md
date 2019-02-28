# Computer Science 600 Research Project
-----
### By Nakul Iyer, with help from Dr. Nicholas Zufelt
#### Phillips Academy, February 2019

## Intro
The purpose of this project was to create an automatic music transcription engine using convolutional neural networks. Commentary on the successes and failures of the project can be found in `documentation.pdf`

## Background Knowledge
Constant-Q Transform, Signal Processing, Non-Negative Matrix Factorization, Convolutional Neural Networks, Spectrograms, Tensorflow/Keras, Short-Time Fourier Transform (See `bibliography.pdf` and `documentation.pdf`)

## Dataset
MAPS (Midi-Aligned Piano Sounds)

## How to Run
The following python modules are necessary and should be installed at the newest version through conda or pip:
* NumPy
* SciPy
* Tensorflow
* Keras
* MatPlotLib
Alternatively, run `setup.py` to set up all these dependencies if you do not already have them. If you opt to use a GPU with Tensorflow, install Tensorflow with GPU enabled and make sure to allow your GPU to be visible to Tensorflow in your PATH variable (different for each GPU, follow instructions for your specific GPU)

IMPORTANT: The dependency LilyPond is vital for downloading sheet music as a pdf. To install Lilypond, follow the instructions on the website [http://lilypond.org](http://lilypond.org). Essentially, you will need to download the program and run the installation file attached to the installer. Then, put the `/bin` for LilyPond in your PATH variable. For Windows, this path is `C:\Program Files (x86)\LilyPond\usr\bin`. Then, be sure to test in terminal if the command `lilypond` works. If this dependency is not installed, this program will only show notes and their onset/offset times rather than real sheet music.

Finally, the two relevant user-end files are the `src/train.py` and `src/run.py` files. `src/train.py` is meant for training purposes and will initialize a neural network with parameters and architecture defined by `src/config.py` and `src/train.py`. `src/run.py` will predict on a given test file and will place a pdf of the transcribed song in a specified output location. The two commands should be run as follows:
* `python train.py`
* `python run.py C:/path/to/song.wav C:/path/to/output/save/location.pdf`

Also, if you are doing training, change the dataset directory path `mus_path` inside the `train.py` file to the path of the MAPS dataset on your local system.

## Thank you for using Sheet Music Helper!
Please send any questions about this product to nakulpiyer@gmail.com. Visit our website at https://nakuliyer.github.io/music-transcription-engine/site/index.html! For further reading, please see my `bibliography.pdf` file.
