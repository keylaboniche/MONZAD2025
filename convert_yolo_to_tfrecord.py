import os
import tensorflow as tf
from object_detection.utils import dataset_util
from PIL import Image

# === CONFIGURACIÓN DE CLASES ===
LABEL_DICT = {0: 'pollo'}  # SOLO UNA CLASE
IMAGE_FORMAT = b'jpeg'  # usa 'jpeg' o 'png' según tus imágenes

def create_tf_example(image_path, label_path):
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()
    image = Image.open(image_path)
    width, height = image.size

    filename = os.path.basename(image_path).encode('utf8')
    image_format = IMAGE_FORMAT

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, box_width, box_height = map(float, parts)
            xmins.append(x_center - box_width / 2)
            xmaxs.append(x_center + box_width / 2)
            ymins.append(y_center - box_height / 2)
            ymaxs.append(y_center + box_height / 2)
            classes_text.append(LABEL_DICT[int(class_id)].encode('utf8'))
            classes.append(int(class_id) + 1)  # TF empieza en 1

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def generate_tfrecord(output_path, images_dir, labels_dir):
    writer = tf.io.TFRecordWriter(output_path)
    for img_name in os.listdir(images_dir):
        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        if not os.path.exists(label_path):
            continue
        tf_example = create_tf_example(image_path, label_path)
        writer.write(tf_example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    generate_tfrecord('annotations/train.record', 'data/images/train', 'data/labels/train')
    generate_tfrecord('annotations/val.record', 'data/images/val', 'data/labels/val')
