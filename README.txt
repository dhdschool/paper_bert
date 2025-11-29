This package was written for WSL2 Ubuntu and is not guaranteed to work on
every system, but should run on most systems that support cuda libraries 
and CPython. It also requires a system with >4GB of memory, preferably VRAM on
an nvidia GPU but this isn't strictly necessary.

We recommend using conda miniforge3 for virtual environment management, but venv
is suitable.

Once you have a suitable python interpreter and the latest version of pip,
navigate to this directory and run "pip install -r requirements.txt" to install
the package requirements

To run the package, navigate to this directory and run "python main.py".
This will download and process the dataset (with help from the data.py file),
then begin fine tuning the model (with help from the training.py file).
This will create new subdirectories, dataset and model, containing the kaggle data
and model training weights, respectively. Once the model is done training, the
kaggle test output will be evaluated and placed in the dataset directory.

Model metrics that are calculated over the evaluation set can be found in the
model/checkpoint-(training steps)/trainer_state.json at the bottom, around line
280.

