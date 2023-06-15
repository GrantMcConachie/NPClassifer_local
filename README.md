# NPClassifer_local
This app is a local version of NPClassifier: https://github.com/mwang87/NP-Classifier <br><br>
Instead of having neural networks trained then hosted on UCSD servers, the neural networks are trained and stored in the 'trained_models' folder. This version of the app also has a 'batch processing' section so you can upload excel files and do multiple classifications at a time without needing to script.<br><br>

This neural network is not as wholistic at the actual NPclassifier neural network. The data that it was trained on is different from theirs, however the 'class' classification still has ~72% similarity with NPclassifier. The 'pathway' and 'superclass' classifications are ~100% similar with NPclassifier. 

## Instructions
```shell
# clone
git clone https://github.com/GrantMcConachie/NPClassifer_local.git

# cd to directory
cd NPClassifier_local

# unzip zipped folders
python -m venv venv
cd venv/Scripts
activate

# install libraries
pip install -r requirements.txt

# Start app
python app.py
```
The app will run locally at http://127.0.0.1:8080/
