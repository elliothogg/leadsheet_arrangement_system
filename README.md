# Leadsheet Arrangement System (LSAS)

This project is a masters dissertation that uses deep learning to generate full arrangements of jazz piano leadsheets.

The project is supervised by:  

David Herbert  

Mark Turner


## Components

The LSAS is made up of 3 main components:

- `chord_scraper`: This is a script than can extract chord symbol-chord voicing pairs from any fully arranged piano score with annoted chord symbols in MusicXML format

- `chord_generator`: This features a cDCGAN model capable of generated chord voicings for a given chord label. It also includes some other deep learning experiments

- `leadsheet_arranger`: This is the system that performs lead sheet arrangement. It takes MusicXML lead sheets and outputs full arrangements of the same format


## Paper

[Please read for more information](https://github.com/elliothogg/leadsheet_arrangement_system/blob/master/paper/Leadsheet%20Arrangement%20Using%20Deep%20Learning.pdf)


## Usage 

Each component has its own README; please see each one for more information.

Python version = `3.8.3`

You must have venv installed globally

You must also have graphviz installed globally for Keras model plots to work



### Install:

```
cd leadsheet_arrangement_system
python3 -m venv venv/
source venv/bin/activate
pip install -r requirements.txt
```

#### To exit the virtual environment:
```
deactivate
```
#### To reactivate the virtual environment:
```
source venv/bin/activate
```
### Vscode

If you're using Vscode, make sure that your interpreter is set to `"./venv/bin/python"` or Python 3.8.3 64-bit (`'venv':venv`).

You may have to set Python: Venv Path (in settings) to "~/venv"
