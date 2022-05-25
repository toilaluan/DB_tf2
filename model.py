from turtle import shape
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, models
import losses
def dbnet(input_size = 640, k =50):
    image_input = layers.Input(shape=(None,None,3))
    gt_input = layers.Input(shape=(input_size, input_size))
    mask_input = layers.Input(shape=(input_size, input_size))
    thresh_input = layers.Input(shape=(input_size, input_size))
    thresh_mask_input = layers.Input(shape=(input_size, input_size))
    backbone = ResNet50(input_tensor = image_input, include_top = False)
    backbone.summary()
    C5 = backbone.get_layer('conv5_block3_out').output
    C4 = backbone.get_layer('conv4_block6_out').output
    C3 = backbone.get_layer('conv3_block4_out').output
    C2 = backbone.get_layer('conv2_block3_out').output
    in2 = layers.Conv2D(256, 1, padding='same', name='in2', use_bias=False)(C2)
    in3 = layers.Conv2D(256, 1, padding='same', name='in3', use_bias=False)(C3)
    in4 = layers.Conv2D(256, 1, padding='same', name='in4', use_bias=False)(C4)
    in5 = layers.Conv2D(256, 1, padding='same', name='in5', use_bias=False)(C5)

    P5 = layers.Conv2DTranspose(64, 8, 8, name='P5', use_bias=False)(in5)
    P4 = layers.Conv2DTranspose(64, 4, 4, name='P4', use_bias=False)(in4)
    P3 = layers.Conv2DTranspose(64, 2, 2, name='P3', use_bias=False)(in3)
    P2 = layers.Conv2DTranspose(64, 1, 1, name='P2', use_bias=False)(in2)
    fuse = layers.Concatenate()([P2, P3, P4, P5])

    # Probability map
    p = layers.Conv2D(64, 3, padding='same', use_bias=False)(fuse)
    p = layers.BatchNormalization()(p)
    p = layers.ReLU()(p)
    p = layers.Conv2DTranspose(64, 2, 2, use_bias=False)(p)
    p = layers.BatchNormalization()(p)
    p = layers.ReLU()(p)
    p = layers.Conv2DTranspose(1, 2, 2, use_bias=False, activation = 'sigmoid')(p)

    # Threshold map 
    t = layers.Conv2D(64, 3, padding='same', use_bias=False)(fuse)
    t = layers.BatchNormalization()(t)
    t = layers.ReLU()(t)
    t = layers.Conv2DTranspose(64, 2, 2, use_bias=False)(t)
    t = layers.BatchNormalization()(t)
    t = layers.ReLU()(t)
    t = layers.Conv2DTranspose(1, 2, 2, use_bias=False, activation = 'sigmoid')(t)

    # Approx Binary Map
    b_hat = layers.Lambda(lambda x: 1 / (1+tf.exp(-k*(x[0]-x[1]))))([p, t])

    loss = layers.Lambda(losses.db_loss, name='db_loss')([p, b_hat, gt_input, mask_input, t, thresh_input, thresh_mask_input])
    training_model = models.Model(inputs=[image_input, gt_input, mask_input, thresh_input, thresh_mask_input], outputs=loss)
    prediction_model = models.Model(inputs=image_input, outputs = p)
    return training_model, prediction_model
if __name__ == '__main__':
    model, predict_model = dbnet()