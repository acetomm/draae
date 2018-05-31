import numpy as np
import tensorflow as tf

X_DIM = 784

C_DIM = 50
S_DIM = 50

MODEL_FILE = 'model/model.ckpt'

def e(x):
    if hasattr(e, 'reuse'):
        e.reuse = True
    else:
        e.reuse = False
    h1 = tf.layers.dense(x, 500, tf.nn.relu, name='e_h1', reuse=e.reuse)
    h2 = tf.layers.dense(h1, 400, tf.nn.relu, name='e_h2', reuse=e.reuse)
    return h2

def ec(h):
    if hasattr(ec, 'reuse'):
        ec.reuse = True
    else:
        ec.reuse = False
    h1 = tf.layers.dense(h, 300, tf.nn.relu, name='ec_h1', reuse=ec.reuse)
    h2 = tf.layers.dense(h1, 150, tf.nn.relu, name='ec_h2', reuse=ec.reuse)
    c = tf.layers.dense(h2, 100, tf.nn.relu, name='ec_c', reuse=ec.reuse)
    return c

def ec_tail(c):
    if hasattr(ec_tail, 'reuse'):
        ec_tail.reuse = True
    else:
        ec_tail.reuse = False
    h1 = tf.layers.dense(c, 300, tf.nn.relu, name='e_c_h1', reuse=ec_tail.reuse)
    h2 = tf.layers.dense(h1, 200, tf.nn.relu, name='e_c_h2', reuse=ec_tail.reuse)
    c_tail = tf.layers.dense(h2, C_DIM, None, name='e_c_h3', reuse=ec_tail.reuse)
    return c_tail
    
    
def es(h):
    if hasattr(es, 'reuse'):
        es.reuse = True
    else:
        es.reuse = False
    h1 = tf.layers.dense(h, 100, tf.nn.relu, name='es_h1', reuse=es.reuse)
    s = tf.layers.dense(h1, S_DIM, None, name='es_s', reuse=es.reuse)
    return s


def d(cs):
    if hasattr(d, 'reuse'):
        d.reuse = True
    else:
        d.reuse = False
    h1 = tf.layers.dense(cs, 200, tf.nn.relu, name='d_h1', reuse=d.reuse)
    h2 = tf.layers.dense(h1, 500, tf.nn.relu, name='d_h2', reuse=d.reuse)
    xr = tf.layers.dense(h2, X_DIM, tf.nn.sigmoid, name='d_xr', reuse=d.reuse)
    return xr

def sdc(s):
    if hasattr(sdc, 'reuse'):
        sdc.reuse = True
    else:
        sdc.reuse = False
    h1 = tf.layers.dense(s, 200, tf.nn.relu, name='sdc_h1', reuse=sdc.reuse)
    h2 = tf.layers.dense(h1, 100, tf.nn.relu, name='sdc_h2', reuse=sdc.reuse)
    logits = tf.layers.dense(h2, 1, None, name='sdc_logits', reuse=sdc.reuse)
    return logits

def rs(batch_size):
    return np.random.randn(batch_size, S_DIM) * 1.

def cdc(c):
    if hasattr(cdc, 'reuse'):
        cdc.reuse = True
    else:
        cdc.reuse = False
    h1 = tf.layers.dense(c, 200, tf.nn.relu, name='cdc_h1', reuse=cdc.reuse)
    h2 = tf.layers.dense(h1, 100, tf.nn.relu, name='cdc_h2', reuse=cdc.reuse)
    logits = tf.layers.dense(h2, 1, None, name='cdc_logits', reuse=cdc.reuse)
    return logits

def rc(batch_size):
    return np.random.randn(batch_size, C_DIM) * 1.


