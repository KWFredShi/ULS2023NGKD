import os
import pickle
import random
import nibabel as nib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, BatchNormalization, Activation

BONE_DATA_PATH = '/nobackup/cs3892-oguz/wangjl3/ULS23_Radboudumc_Bone/images'
BONE_ANNOTATION_PATH = '/nobackup/cs3892-oguz/wangjl3/ULS23_Radboudumc_Bone/labels'
# PANCREAS_DATA_PATH = '/content/drive/MyDrive/Junior/Machine_Learning_Project/Data/NGKDStorage/images/ULS23_Part1/ULS23/novel_data/ULS23_Radboudumc_Pancreas/images'
# PANCREAS_ANNOTATION_PATH = '/content/drive/MyDrive/Junior/Machine_Learning_Project/Data/NGKDStorage/annotations/ULS23/novel_data/ULS23_Radboudumc_Pancreas/labels'
DIMENSION = (256, 256, 128, 1)
MODEL_PATH = '/home/shik2/TEAM/ULS23/Finn/deeplabv3+/unet.h5'
BATCH_SIZE = 32 
NUM_EPOCH = 1 
PATCH_SIZE = (64, 64, 32)
HISTORY_PATH = '/home/shik2/TEAM/ULS23/Finn/deeplabv3+/trainHistory'
NUM_FILES = 3 

class MultiVolumePatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, volumes, labels, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE, shuffle=True):
        self.volumes = volumes  # List of volumes
        self.labels = labels  # Corresponding list of label volumes
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = self._get_indexes()
        self.on_epoch_end()

    def _get_indexes(self):
        # Create an index for each patch in each volume
        indexes = []
        for volume_index, volume in enumerate(self.volumes):
            num_patches_x = volume.shape[0] // self.patch_size[0]
            num_patches_y = volume.shape[1] // self.patch_size[1]
            num_patches_z = volume.shape[2] // self.patch_size[2]
            num_patches = num_patches_x * num_patches_y * num_patches_z
            for patch_index in range(num_patches):
                indexes.append((volume_index, patch_index))
        return indexes

    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self._generate_data(batch_indexes)
        return X, y

    def on_epoch_end(self):
        # Shuffle indexes for the next epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_data(self, batch_indexes):
        X = np.empty((len(batch_indexes), *self.patch_size, self.volumes[0].shape[-1]), dtype=np.float32)
        y = np.empty((len(batch_indexes), *self.patch_size, self.labels[0].shape[-1]), dtype=np.float32)

        for i, (volume_index, patch_index) in enumerate(batch_indexes):
            volume = self.volumes[volume_index]
            label = self.labels[volume_index]

            num_patches_x = volume.shape[0] // self.patch_size[0]
            num_patches_y = volume.shape[1] // self.patch_size[1]

            patch_idx_x = (patch_index % num_patches_x) * self.patch_size[0]
            patch_idx_y = ((patch_index // num_patches_x) % num_patches_y) * self.patch_size[1]
            patch_idx_z = (patch_index // (num_patches_x * num_patches_y)) * self.patch_size[2]

            X[i,] = volume[patch_idx_x:patch_idx_x+self.patch_size[0],
                           patch_idx_y:patch_idx_y+self.patch_size[1],
                           patch_idx_z:patch_idx_z+self.patch_size[2], :]
            y[i,] = label[patch_idx_x:patch_idx_x+self.patch_size[0],
                          patch_idx_y:patch_idx_y+self.patch_size[1],
                          patch_idx_z:patch_idx_z+self.patch_size[2], :]
        return X, y

def randomPick(data_path, annotation_path):
  all_files = os.listdir(data_path)
  random_filename = random.choice(all_files)

  data_file = os.path.join(data_path, random_filename)
  data_file = nib.load(data_file).get_fdata()

  annotation_file = os.path.join(annotation_path, random_filename)
  annotation_file = nib.load(annotation_file).get_fdata()
  return data_file, annotation_file

def generateData(numData=NUM_FILES):
  X_train = []
  Y_train = []

  for i in range(numData):
    x, y = randomPick(BONE_DATA_PATH, BONE_ANNOTATION_PATH)
    X_train.append(x)
    Y_train.append(y)

  X_train = np.array(X_train)
  Y_train = np.array(Y_train)

  return X_train, Y_train

def conv_block(input_tensor, num_filters):
    x = Conv3D(num_filters, (3, 3, 3), padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv3D(num_filters, (3, 3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def up_conv_block(input_tensor, skip_tensor, num_filters):
    x = UpSampling3D((2, 2, 2))(input_tensor)
    x = concatenate([x, skip_tensor], axis=-1)
    x = conv_block(x, num_filters)
    return x

def unet_3d(input_shape=(64,64,32,1), num_classes=1):
    inputs = Input(input_shape)

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = conv_block(p1, 64)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = conv_block(p2, 128)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = conv_block(p3, 256)
    p4 = MaxPooling3D((2, 2, 2))(c4)

    # Bottleneck
    b = conv_block(p4, 512)  # Adjusted to be the bottleneck after adding a new layer

    # Decoder
    u1 = up_conv_block(b, c4, 256)
    u2 = up_conv_block(u1, c3, 128)
    u3 = up_conv_block(u2, c2, 64)
    u4 = up_conv_block(u3, c1, 32)

    # Output layer
    outputs = Conv3D(num_classes, (1, 1, 1), activation='sigmoid')(u4)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

# Define your dice_score function here
def dice_score(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# Dice loss, calculated as 1 - dice_score
def dice_loss(y_true, y_pred):
    return 1 - dice_score(y_true, y_pred)

def loadModel(model_path):
    if os.path.exists(model_path):
        print("Loading existing model.")
        model = load_model(model_path, custom_objects={'dice_loss': dice_loss, 'dice_score': dice_score})
    else:
        print("Creating a new model.")
        model = unet_3d()
        
        model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_score])
        
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    return model

X_train, Y_train = generateData()
model = loadModel(MODEL_PATH)
generator = MultiVolumePatchGenerator(X_train, Y_train)
history = model.fit(generator, epochs=NUM_EPOCH)
model.save(MODEL_PATH)
with open(HISTORY_PATH, 'wb') as historyFile:
   pickle.dump(history.history, historyFile)

