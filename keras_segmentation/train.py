import json
import os

from .data_utils.data_loader import image_segmentation_generator, \
    verify_segmentation_dataset
import six
from keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import glob
import sys

def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    # This is legacy code, there should always be a "checkpoint" file in your directory

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    all_checkpoint_files = [ff.replace(".index", "") for ff in
                            all_checkpoint_files]  # to make it work for newer versions of keras
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f)
                                       .isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid"
                             .format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files,
                                  key=lambda f:
                                  int(get_epoch_number_from_path(f)))

    return latest_epoch_checkpoint

def masked_categorical_crossentropy(gt, pr):
    from keras.losses import categorical_crossentropy
    mask = 1 - gt[:, :, 0] 
    return categorical_crossentropy(gt, pr) * mask

def weighted_categorical_crossentropy(class_weights):
    from keras.losses import categorical_crossentropy
    class_weights = list(class_weights.values())
    
    def loss(gt, pr):
        _, n_output_pixels, n_class= pr.shape
        
        weights_tensor = tf.repeat(class_weights, n_output_pixels)
        weights_tensor = tf.reshape(weights_tensor, (-1, n_output_pixels, n_class))
        weights = tf.reduce_sum(gt * weights_tensor, axis=-1)

        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        loss = cce(gt, pr)
        weighted_loss = loss * weights
        print(weighted_loss.shape)
        return tf.reduce_mean(weighted_loss)
    return loss


def helper_calc_lls(softmax_mask):
    mask = tf.one_hot(tf.argmax(softmax_mask, 2), 3)
    l_coord = tf.where(mask[:,:,1] == 1)
    r_coord = tf.where(mask[:,:,2] == 1)

    # get all coords of image - will be used later
    coord_map = tf.dtypes.cast(tf.where(mask[:,:,-1] == mask[:,:,-1]), tf.float32)
    mask_flat = tf.reshape(mask, (-1, 3))

    try:
        l_y = tf.dtypes.cast(tf.reshape(l_coord[:, 0], (-1, 1)), tf.float32)
        l_Y = tf.concat([l_y, tf.reshape(tf.ones_like(l_y), (-1, 1))], axis=1)
        #tf.print("l_Y", l_Y, len(l_y))
        l_x = tf.dtypes.cast(l_coord[:, 1], tf.float32)
        l_X = tf.reshape(l_x, (-1, 1))
        #tf.print("l_X", l_X, len(l_X))
        try:
            if len(l_y) > 10:
                l_params = tf.linalg.lstsq(l_Y, l_X, fast=False)
            else:
                l_params = tf.constant([[0], [0]], dtype=tf.float32)
        except Exception as e:
            tf.print("Error in l_lls calc",e,l_Y, l_X)
            l_params = tf.constant([[0], [0]], dtype=tf.float32)
        #tf.print("lparams", l_params)

        r_y = tf.dtypes.cast(tf.reshape(r_coord[:, 0], (-1, 1)), tf.float32)
        r_Y = tf.concat([r_y, tf.reshape(tf.ones_like(r_y), (-1, 1))], axis=1)
        r_x = tf.dtypes.cast(r_coord[:, 1], tf.float32)
        r_X = tf.reshape(r_x, (-1, 1))
        #tf.print("r_Y", r_Y)
        #tf.print("r_X", r_X)
        try:
            if len(r_y) > 10:
                r_params = tf.linalg.lstsq(r_Y, r_X, fast=False)
            else:
                r_params = tf.constant([[0],[0]], dtype=tf.float32)
        except Exception as e:
            tf.print("Error in r_lls calc",r_Y, r_X, e)
            r_params = tf.constant([[0],[0]], dtype=tf.float32)
        #tf.print("rparams", r_params)
        
        params = tf.concat([l_params, r_params], axis=0)
        #tf.print("params", params)
        
        # compute l and r lane masks
        l_error = (coord_map[:,1] - (coord_map[:,0]*l_params[0] + l_params[1]))**2
        l_error_mask = l_error * mask_flat[:,1]
        r_error = (coord_map[:,1] - (coord_map[:,0]*r_params[0] + r_params[1]))**2
        r_error_mask = r_error * mask_flat[:,2]
        
        # stack background error mask, left lane error mask, right lane error mask
        error_mask = tf.stack([tf.zeros(mask_flat[:,0].shape), l_error_mask, r_error_mask], axis=1)
        #tf.print("Error mask mean", tf.reduce_mean(tf.reduce_mean(error_mask, 0), 0))
        return error_mask
    except Exception as e:
        tf.print("l_coord and r_coord empty", e)
        return tf.zeros(mask_flat.shape)


def custom_lss_loss(output_h, output_w):
    def loss(gt, pr):
        batch_size, n_output_pixels, n_class= pr.shape
        assert n_output_pixels == output_h * output_w    
        # reshape the flat pr to pr mask of image shape
        pr_mask = tf.reshape(pr, (-1, output_h, output_w, n_class))
        # calculate the error masks
        error_mask = tf.map_fn(fn=helper_calc_lls, elems = pr_mask, fn_output_signature=tf.float32)
        #tf.print(" Shape of error mask", error_mask.shape)
        return tf.reduce_mean(error_mask)
    loss.__name__ = 'custom_lls_loss'
    return loss

def joint_ce_shape_loss(output_h, output_w, shape_loss_weight = 0.3):
    def loss(gt, pr):
        part_crossentropy = 1 - shape_loss_weight
        part_custom = shape_loss_weight
        # crossentropy
        loss_categorical_crossentropy = tf.keras.losses.categorical_crossentropy(gt, pr)
        # custom_loss
        loss_lls = custom_lss_loss(output_h, output_w)(gt, pr)
        if loss_lls > 0:
            return part_crossentropy*loss_categorical_crossentropy + part_custom*loss_lls
        else:
            return loss_categorical_crossentropy
    loss.__name__ = 'joint_ce_shape_loss'
    return loss

def custom_contrastive_loss(batch_size, n_contrastive):
    n_instances_per_img = n_contrastive + 1
    def loss(gt, pr):
        error = 0
        n = 0
        for i in range(0, batch_size, n_instances_per_img):
            for a in range(n_instances_per_img):
                for b in range(a+1, n_instances_per_img):
                    img_a = tf.argmax(pr[i+a], 1)
                    img_b = tf.argmax(pr[i+b], 1)
                error += tf.math.reduce_sum(tf.cast(img_a != img_b, tf.float32))/pr.shape[1]
                n += 1
        return tf.math.sqrt(error/n)
    loss.__name__ = "cont_loss"
    return loss


def joint_ce_cont_loss(batch_size, n_contrastive, contrastive_loss_weight=0.3):
    def loss(gt, pr):
        part_crossentropy = 1 - contrastive_loss_weight
        part_custom = contrastive_loss_weight
        # crossentropy
        loss_categorical_crossentropy = tf.keras.losses.categorical_crossentropy(gt, pr)
        # custom_loss
        loss_contrastive = custom_contrastive_loss(batch_size, n_contrastive)(gt, pr)
        if loss_contrastive > 0:
            return part_crossentropy*loss_categorical_crossentropy + part_custom*loss_contrastive
        else:
            return loss_categorical_crossentropy
    loss.__name__ = 'joint_ce_cont_loss'
    return loss


class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoints_path is not None:
            self.model.save_weights(self.checkpoints_path + "." + str(epoch))
            print("saved ", self.checkpoints_path + "." + str(epoch))


def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          val_steps_per_epoch=512,
          gen_use_multiprocessing=False,
          ignore_zero_class=False,
          optimizer_name='adam',
          do_augment=False,
          augmentation_name="aug_all",
          callbacks=None,
          custom_augmentation=None,
          do_contrastive=False,
          custom_contrastives = None,
          contrastive_loss_weight = 0.3,
          other_inputs_paths = None,
          preprocessing=None,
          class_weights = None,
          do_shape = False,
          shape_loss_weight = 0.3,
          read_image_type=1  # cv2.IMREAD_COLOR = 1 (rgb),
                             # cv2.IMREAD_GRAYSCALE = 0,
                             # cv2.IMREAD_UNCHANGED = -1 (4 channels like RGBA)
         ):
    from .models.all_models import model_from_name
    # check if user gives model name instead of the model object
    if isinstance(model, six.string_types):
        # create the model from the name
        assert (n_classes is not None), "Please provide the n_classes"
        if (input_height is not None) and (input_width is not None):
            model = model_from_name[model](
                n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert val_images is not None
        assert val_annotations is not None

    if optimizer_name is not None:

        #if ignore_zero_class:
        if class_weights:
            model.compile(loss=weighted_categorical_crossentropy(class_weights), 
                    optimizer=optimizer_name, metrics=['accuracy'])
        elif do_shape:
            model.compile(loss = joint_ce_shape_loss(output_height, output_width, shape_loss_weight),
                    optimizer=optimizer_name, 
                    metrics=['accuracy', "categorical_crossentropy", 
                    custom_lss_loss(output_height, output_width)])
        elif do_contrastive:
            n_contrastive = len(custom_contrastives)
            model.compile(loss=joint_ce_cont_loss(batch_size, n_contrastive, contrastive_loss_weight),
                    optimizer=optimizer_name, 
                    metrics=[ 'categorical_crossentropy', 
                    custom_contrastive_loss(batch_size, n_contrastive), 'acc'])
        else:
            loss = 'categorical_crossentropy'
            model.compile(loss=loss, optimizer=optimizer_name,metrics=['accuracy'])

    if checkpoints_path is not None:
        config_file = checkpoints_path + "_config.json"
        dir_name = os.path.dirname(config_file)

        if ( not os.path.exists(dir_name) )  and len( dir_name ) > 0 :
            os.makedirs(dir_name)

        with open(config_file, "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    initial_epoch = 0

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)
            initial_epoch = int(latest_checkpoint.split('.')[-1])

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images,
                                               train_annotations,
                                               n_classes)
        assert verified
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images,
                                                   val_annotations,
                                                   n_classes)
            assert verified

    train_gen = image_segmentation_generator(
        train_images, train_annotations,  batch_size,  n_classes,
        input_height, input_width, output_height, output_width,
        do_augment=do_augment, augmentation_name=augmentation_name,
        custom_augmentation = custom_augmentation, 
        do_contrastive = do_contrastive,
        custom_contrastives = custom_contrastives,
        other_inputs_paths=other_inputs_paths, 
        preprocessing=preprocessing, read_image_type=read_image_type)
    
    if validate:
        val_gen = image_segmentation_generator(
                val_images, val_annotations,  val_batch_size,
                n_classes, input_height, input_width, output_height, output_width,
                other_inputs_paths=other_inputs_paths,
                preprocessing=preprocessing, read_image_type=read_image_type)
    

    if callbacks is None and (not checkpoints_path is None):
        default_callback = ModelCheckpoint(
                filepath=checkpoints_path + ".{epoch:05d}",
                save_weights_only=True,
                verbose=True
            )
        if sys.version_info[0] < 3:
            default_callback = CheckpointsCallback(checkpoints_path)

    
    callbacks = [default_callback]
    
    if not validate:
        history = model.fit(train_gen, steps_per_epoch=steps_per_epoch,
                  epochs=epochs, callbacks=callbacks, initial_epoch=initial_epoch)
    else:
        history = model.fit(train_gen,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=val_gen,
                  validation_steps=val_steps_per_epoch,
                  epochs=epochs, callbacks=callbacks,
                  use_multiprocessing=gen_use_multiprocessing, 
                  initial_epoch=initial_epoch)
    return history
