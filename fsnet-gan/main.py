import os
import tensorflow.compat.v1 as tf
import numpy as np
import random

from ini_file_io import load_train_ini
from model import drnet_3D_xy

# set cuda visable device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def main(_):
    tf.reset_default_graph()
    # load training parameter #
    ini_file = 'tr_param.ini'
    param_sets = load_train_ini(ini_file)
    param_set = param_sets[0]
    
    set_random_seed(0)

    print ('====== Phase >>> %s <<< ======' % param_set['phase'])

    if not os.path.exists(param_set['chkpoint_dir']):
        os.makedirs(param_set['chkpoint_dir'])

    if not os.path.exists(param_set['results_dir']):
        os.makedirs(param_set['results_dir'])

    if not os.path.exists(param_set['labeling_dir']):
        os.makedirs(param_set['labeling_dir'])

    # GPU setting, per_process_gpu_memory_fraction means 95% GPU MEM ,allow_growth means unfixed memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        model = drnet_3D_xy(sess, param_set)

        if param_set['phase'] == 'train':
            model.train()
        elif param_set['phase'] == 'test':
            model.test(param_set['epoch'])
            

if __name__ == '__main__':
    tf.app.run()
