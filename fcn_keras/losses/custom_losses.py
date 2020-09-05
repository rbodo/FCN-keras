import tensorflow as tf
import keras.backend as k


def custom_categorical_crossentropy():
    """
    Create a loss function that adds the MSE loss to the mean of all squared
    activations of a specific layer.
    """

    def loss(y_true, y_pred):
        # axis = 3 is softmax
        loss_val = k.categorical_crossentropy(y_true, y_pred,
                                              from_logits=False, axis=3)

        mean_loss = tf.math.reduce_mean(loss_val)

        return mean_loss

    # Return a function
    return loss
