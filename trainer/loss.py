from tensorflow.keras import backend as K
import tensorflow as tf

_EPSILON = K.epsilon()


def hinge_loss_fn(batch_size):
    def hinge_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        loss = tf.convert_to_tensor(0,dtype=tf.float32)
        g = tf.constant(1.0, shape=[1], dtype=tf.float32)
        for i in range(0, batch_size, 3):
            try:
                q_embedding = y_pred[i+0]
                p_embedding = y_pred[i+1]
                n_embedding = y_pred[i+2]
                D_q_p =  K.sqrt(K.sum((q_embedding - p_embedding)**2))
                D_q_n = K.sqrt(K.sum((q_embedding - n_embedding)**2))
                loss = (loss + g + D_q_p - D_q_n)
            except:
                continue
        loss = loss/(batch_size/3)
        zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
        return tf.maximum(loss,zero)
    return hinge_loss


def hinge_new_loss_fn(batch_size):
    def hinge_new_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        loss = tf.convert_to_tensor(0,dtype=tf.float32)
        g = tf.constant(1.0, shape=[1], dtype=tf.float32)
        for i in range(0, batch_size, 3):
            try:
                q_embedding = y_pred[i+0]
                p_embedding = y_pred[i+1]
                n_embedding = y_pred[i+2]
                D_q_p =  K.sqrt(K.sum((q_embedding - p_embedding)**2))
                D_q_n = K.sqrt(K.sum((q_embedding - n_embedding)**2))
                D_p_n = K.sqrt(K.sum((p_embedding - n_embedding)**2))
                loss = (loss + g + D_q_p - D_q_n + D_q_p - D_p_n)
            except:
                continue
        loss = loss/(batch_size/6)
        zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
        return tf.maximum(loss,zero)
    return hinge_new_loss


def hinge_twice_loss_fn(batch_size):
    def hinge_twice_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        loss = tf.convert_to_tensor(0,dtype=tf.float32)
        g = tf.constant(1.0, shape=[1], dtype=tf.float32)
        for i in range(0, batch_size, 3):
            try:
                q_embedding = y_pred[i+0]
                p_embedding = y_pred[i+1]
                n_embedding = y_pred[i+2]
                D_q_p =  K.sqrt(K.sum((q_embedding - p_embedding)**2))
                D_q_n = K.sqrt(K.sum((q_embedding - n_embedding)**2))
                loss = (loss + g + D_q_p - D_q_n)
            except:
                continue
        loss = loss/(batch_size/6)
        zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
        return tf.maximum(loss,zero)
    return hinge_twice_loss


def contrastive_loss_fn(batch_size):
    def contrastive_loss(y_true, y_pred):
        def _contrastive_loss(y1, D):
            g = tf.constant(1.0, shape=[1], dtype=tf.float32)
            return K.mean(y1 * K.square(D) +
                          (g - y1) * K.square(K.maximum(g - D, 0)))

        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        loss = tf.convert_to_tensor(0,dtype=tf.float32)
        g = tf.constant(1.0, shape=[1], dtype=tf.float32)
        h = tf.constant(0.0, shape=[1], dtype=tf.float32)
        for i in range(0,batch_size,3):
            try:
                q_embedding = y_pred[i+0]
                p_embedding = y_pred[i+1]
                n_embedding = y_pred[i+2]
                D_q_p = K.sqrt(K.sum((q_embedding - p_embedding)**2))
                D_q_n = K.sqrt(K.sum((q_embedding - n_embedding)**2))
                L_q_p = _contrastive_loss(g, D_q_p)
                L_q_n = _contrastive_loss(h, D_q_n)
                loss = (loss + L_q_p + L_q_n )
            except:
                continue
        loss = loss/(batch_size*2/3)
        zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
        return tf.maximum(loss,zero)
    return contrastive_loss

