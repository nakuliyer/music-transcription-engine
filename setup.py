import os

requirements_path = 'requirements.txt'
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()
setup(name="SheetMusicHelper",
      version="1.0",
      description="Automatic Music Transcription Engine using Convolutional Neural Networks in Python",
      author="Nakul Iyer",
      author_email="nakulpiyer@gmail.com",
      install_requires=install_requires)
