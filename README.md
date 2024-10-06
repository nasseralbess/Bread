# Bread
Seismic events detection for the Nasa Space Apps challenge
Using autoencoder deep learining architecture for anomaly detection, utilizing the tensorflow and keras libraries.

# For our model's evaluation of the test set:
Go into the catalogues directory, which contains two separate subfolders, one for the lunar data, and one for the mars data.
We've formatted the catalogues so they follow the format in the lunar and mars training data.

# For the web app
in the terminal:
```console
foo@bar:~$ python streamlit_app.py
```

# For evaluating the saved pretrained model on all lunar training data recursively:
(directories and paths are editable in the file)
```console
foo@bar:~$ python inference.py
```

# For the simplest data preprocessing and model training approach:
```console
foo@bar:~$ python model.py
```

# Finally, regarding the Jupyter notebook:
This is just our experimentation and thought process. Fell free to look at it, although no guarantees of it making much sense lol.
