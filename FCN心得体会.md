# FCN心得体会

## 1、训练数据准备

修改convert_fcn_dataset.py代码，生成TFRecord数据。

生成example：dict_to_tf_example()

```python
# 文件名
filename = data.split('/')[-1].rstrip('.jpg').encode()

feature_dict = {
        'image/height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_data])),
        'image/label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_label])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpg'])),
    }

```

生成tfrecord文件：create_tf_record()

```python
tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
for data,label in file_pars:
    example = dict_to_tf_example(data,label)
    if example != None:
        print(data)
        tfrecord_writer.write(example.SerializeToString())
```

## 2、模型训练日志

![1542025299645](1542025299645.png)









