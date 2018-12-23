# The main aim of this project is to be able to read tfrecords file through tf.data.TFRecordDataset api and be able to pass it to model.fit function.


In `write_read_check_tfrecords` function in tfrecords_utils.py file,
we are writing the data to a tfrecords file.
Then we read back the written data record by record and check whether
the written data is same as original data.


```
def _convert_toexample(im_path, im_arr, label):

    im_shape = im_arr.shape
    im_bytes = im_arr.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
        'im_path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[im_path.encode("utf-8")])),
        'im_arr': tf.train.Feature(bytes_list=tf.train.BytesList(value=[im_bytes])),
        'im_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=im_shape)),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),

    }))

    return example
```

Later on, we use `create_dataset` function to create our Dataset
from the tfrecord file we wrote. The parsing of the records happens
within `_parse_function` which we pass it to the map function of dataset.

Parse function is as follows 

```buildoutcfg

def _parse_function(proto):

    # define your tfrecord again. Remember that you saved your image as a string.

    keys_to_features = {"im_path": tf.FixedLenSequenceFeature([], tf.string, allow_missing=True),
                        "im_shape": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                        "im_arr": tf.FixedLenSequenceFeature([], tf.string, allow_missing=True),
                        "label": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                        }

    # Load one example
    parsed_features = tf.parse_single_example(serialized=proto, features=keys_to_features)

    parsed_features['im_arr'] = parsed_features['im_arr'][0]
    parsed_features['label'] = parsed_features['label'][0]
    parsed_features['im_arr'] = tf.decode_raw(parsed_features['im_arr'], tf.uint8)
    parsed_features['im_arr'] = tf.reshape(parsed_features['im_arr'], parsed_features['im_shape'])

    return parsed_features['im_arr'], parsed_features['label']

```

Finally, we pass the dataset to the model.fit in train_with_datasets.py.


```buildoutcfg

train_dataset  = create_dataset(train_paths)
test_dataset  = create_dataset(test_paths)

model.fit(
        train_dataset.make_one_shot_iterator(),
        steps_per_epoch=5,
        epochs=10,
        shuffle=True,
        validation_data=test_dataset.make_one_shot_iterator(),
        validation_steps=2,
        verbose=1)

```



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
