import logging
import pathlib
import numpy as np
import glob
import tensorflow as tf
import os
import sys
from PIL import Image

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


def write_read_check_tfrecords( data_path, output_path):


    im_paths = glob.glob(os.path.join(data_path, '**/*.jpg'), recursive=True)

    for filename in im_paths:
        print(filename)

    img_width, img_height = 150, 150



    tfrecord_file_name = output_path

    with tf.python_io.TFRecordWriter(tfrecord_file_name, options=tf.python_io.TFRecordCompressionType.ZLIB) as writer:

        for k in range(len(im_paths)):

            print("value of k = ", k)
            print("file being processed = ", im_paths[k])

            im_path = im_paths[k]

            # Load image and convert to RGB numpy array
            im = Image.open(im_path)
            im = im.convert('RGB')
            im  = im.resize((img_height, img_width))
            im_arr = np.asarray(im)

            if "cat" in im_path:
                print("cat present ", im_path)
                label = 0
            else :
                print("dog present ", im_path)
                label = 1

            example = _convert_toexample(im_path, im_arr, label)
            writer.write(example.SerializeToString())


    ## Read back the array and compare

    print("Started the reading of tfrecords to check if the data written is same")
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_file_name,options=tf.python_io.TFRecordCompressionType.ZLIB )

    for string_record in record_iterator:

        example = tf.train.Example()
        example.ParseFromString(string_record)

        im_path = example.features.feature['im_path'].bytes_list.value[0]
        im_path = im_path.decode("utf-8")

        im_arr = example.features.feature['im_arr'].bytes_list.value[0]
        im_shape = example.features.feature['im_shape'].int64_list.value
        label = example.features.feature['label'].int64_list.value[0]


        im_1d = np.fromstring(im_arr, dtype=np.uint8)

        reconstructed_im = im_1d.reshape(im_shape)

        im2 = Image.open(im_path)
        im2 = im2.convert('RGB')
        im2 = im2.resize((img_height, img_width))
        im_arr_original = np.asarray(im2)

        print("Printing the closeness of the the arrays")
        print(np.allclose(reconstructed_im, im_arr_original))


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


def create_dataset(tfrecord_paths):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(tfrecord_paths, compression_type="ZLIB")

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(lambda x: _parse_function(x), num_parallel_calls=1)

    # This dataset will go on forever
    dataset = dataset.repeat()

    # Set the batchsize
    dataset = dataset.batch(1)


    return dataset


def create_tfrecords(project_dir):

    """A small function to test writing and reading a tfrecod file"""

    pathlib.Path(os.path.join(project_dir, 'data/processed/tfrecords/train')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(project_dir, 'data/processed/tfrecords/test')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(project_dir, 'data/processed/tfrecords/validation')).mkdir(parents=True,exist_ok=True)

    train_data_path = 'data/train'
    train_output_path = 'data/processed/tfrecords/train/train.tfrecord'
    write_read_check_tfrecords(train_data_path, train_output_path)

    test_data_path = 'data/validation'
    test_output_path = 'data/processed/tfrecords/validation/validation.tfrecord'
    write_read_check_tfrecords(test_data_path, test_output_path)



if __name__ == '__main__':



    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    print("project directory",project_dir)
    # Create directories if they don't exist

    print("Creating tfrecords...")
    create_tfrecords(project_dir)

