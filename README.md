# The main aim of this project is to be able to read tfrecords file 
through tf.data.TFRecordDataset api and be able to pass it to
model.fit function. #


In `write_read_check_tfrecords` function in tfrecords_utils.py file,
we are writing the data to a tfrecords file.
Then we read back the written data record by record and check whether
the written data is same as original data.

Later on, we use `create_dataset` function to create our Dataset
from the tfrecord file we wrote. The parsing of the records happens
within `_parse_function` which we pass it to the map function of dataset.

Finally, we pass the dataset to the model.fit in train_with_datasets.py.



Steps to run this project :

1] First create a virtual environment using 

virtualenv env

2] Activate the environment

source ./env/bin/activate

3] Install requirements

pip3 install -r requirements.txt

4] Create the tfrecords file and ensure their correctness.

python3 src/tfrecords_utils.py

5] Run the training

python3 src/train_with_datasets.py


Known Issues : 
https://github.com/tensorflow/tensorflow/issues/24520

There is a fix available in the github issue which you can apply 
to the relevant files and the project will run properly.
