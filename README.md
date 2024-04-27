First, ensure that the epinions dataset (linked in the report) is downloaded and installed in a directory called '''data'''. 

# NCF Model

Running the NCF model is as simple as running the '''ncf_ratings_prediction.ipynb''' Jupyter Notebook once all packages are installed. This will run the preprocessing pipeline and then subsequently run the training and testing experiments.

# GraphRec Model
To run the GraphRec model, first ensure that PyG is installed. Then run '''process_trust.py''' and create a csv file containing all the data from the train, validation, and test csv files concatenated together. Then simply run the '''gncf_r.ipynb''' notebook to run all training and testing experiments.
