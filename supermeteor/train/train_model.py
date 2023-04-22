import pixellib
import tensorflow as tf
from pixellib.custom_train import instance_custom_training
print("###################Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

train_maskrcnn = instance_custom_training()
train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes= 3, batch_size = 1)
train_maskrcnn.load_pretrained_model("mask_rcnn_coco.h5")
# Im Ordner drei-Klassen befinden sich die Ordner Train und Test
train_maskrcnn.load_dataset("drei-Klassen")
train_maskrcnn.train_model(num_epochs = 200, augmentation=False,  path_trained_models = "drei-Klassen")