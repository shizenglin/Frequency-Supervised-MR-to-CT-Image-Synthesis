from __future__ import division
import os
import time
from ops import conv3d, conv_bn_relu, deconv_bn_relu, decom_map, global_conv3d, Inception_layer
from utils import load_data_pairs, decompose_vol2cube, compose_cube2vol, get_batch_patches, mae_psnr_ssim
import numpy as np
import tensorflow.compat.v1 as tf
import nibabel as nib

class drnet_3D_xy(object):
    """ Implementation of 3D U-net"""
    def __init__(self, sess, param_set):
        self.sess           = sess
        self.phase          = param_set['phase']
        self.batch_size     = param_set['batch_size']
        self.inputI_length_size    = param_set['inputI_length_size']
        self.inputI_width_size    = param_set['inputI_width_size']
        self.inputI_height_size    = param_set['inputI_height_size']
        self.inputI_chn     = param_set['inputI_chn']
        self.output_chn_syn = param_set['output_chn_syn']
        self.smooth_r = param_set['smooth_r']
        self.traindata_dir  = param_set['traindata_dir']
        self.chkpoint_dir   = param_set['chkpoint_dir']
        self.lr             = param_set['learning_rate']
        self.beta1          = param_set['beta1']
        self.epoch          = param_set['epoch']
        self.model_name     = param_set['model_name']
        self.save_intval    = param_set['save_intval']
        self.testdata_dir   = param_set['testdata_dir']
        self.labeling_dir   = param_set['labeling_dir']
        self.ovlp_ita       = param_set['ovlp_ita']

        self.inputI_size = [self.inputI_length_size,self.inputI_width_size,self.inputI_height_size]
        # build model graph
        self.build_model()

    def l1_loss(self, prediction, ground_truth):
        """
        :param prediction: the current prediction of the ground truth.
        :param ground_truth: the measurement you are approximating with regression.
        :return: mean of the l1 loss across all voxels.
        """
        absolute_residuals = tf.abs(tf.subtract(prediction, ground_truth))
        sum_residuals = tf.reduce_sum(absolute_residuals)
        sum_weights = tf.size(absolute_residuals)
        return tf.truediv(tf.cast(sum_residuals, dtype=tf.float32),
                  tf.cast(sum_weights, dtype=tf.float32))

    # build model graph
    def build_model(self):
        # input
        self.input_I = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.inputI_size[0], self.inputI_size[1], self.inputI_size[2], self.inputI_chn], name='inputI')
        self.input_gt_syn = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.inputI_size[0], self.inputI_size[1], self.inputI_size[2],1], name='target_syn')
        self.input_gt_high = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.inputI_size[0], self.inputI_size[1], self.inputI_size[2],1], name='target_high')
        # probability for classes
        self.pred_prob_high, self.pred_prob_syn = self.unet_3D_model(self.input_I)
        # ========= l1 loss
        self.high_loss = self.l1_loss(self.pred_prob_high, self.input_gt_high)
        self.syn_loss = self.l1_loss(self.pred_prob_syn, self.input_gt_syn)
        
        self.total_loss = self.high_loss + self.syn_loss
        
        # trainable variables
        self.u_vars = tf.trainable_variables()

        # create model saver
        self.saver = tf.train.Saver(max_to_keep=100)
    
    # 3D unet graph
    def unet_3D_model(self, inputI):
        """3D U-net"""
        
        conv1_1 = conv_bn_relu(input=inputI, output_chn=32, name='conv1_1')
        conv1_2 = conv_bn_relu(input=conv1_1, output_chn=32, name='conv1_2')
        pool1 = tf.layers.max_pooling3d(inputs=conv1_2, pool_size=2, strides=2, name='pool1')
        
        conv2_1 = conv_bn_relu(input=pool1, output_chn=64, name='conv2_1')
        conv2_2 = conv_bn_relu(input=conv2_1, output_chn=64, name='conv2_2')
        pool2 = tf.layers.max_pooling3d(inputs=conv2_2, pool_size=2, strides=2, name='pool2')
        
        conv3_1 = conv_bn_relu(input=pool2, output_chn=128, name='conv3_1')
        conv3_2 = conv_bn_relu(input=conv3_1, output_chn=128, name='conv3_2')
        pool3 = tf.layers.max_pooling3d(inputs=conv3_2, pool_size=2, strides=2, name='pool3')      
        
        conv4_1 = conv_bn_relu(input=pool3, output_chn=256, name='conv4_1')
        conv4_2 = conv_bn_relu(input=conv4_1, output_chn=256, name='conv4_2')
        
        deconv1_1 = deconv_bn_relu(input=conv4_2, output_chn=256, name='deconv1_1')
        deconv1_2 = tf.concat([deconv1_1, conv3_2], axis=4, name='deconv1_2')
        deconv1_3 = conv_bn_relu(input=deconv1_2, output_chn=128,  name='deconv1_3')
        deconv1_4 = conv_bn_relu(input=deconv1_3, output_chn=128, name='deconv1_4')
        
        deconv2_1 = deconv_bn_relu(input=deconv1_4, output_chn=128, name='deconv2_1')
        deconv2_2 = tf.concat([deconv2_1, conv2_2], axis=4, name='deconv2_2')
        deconv2_3 = conv_bn_relu(input=deconv2_2, output_chn=64,  name='deconv2_3')
        deconv2_4 = conv_bn_relu(input=deconv2_3, output_chn=64, name='deconv2_4')
        
        deconv3_1 = deconv_bn_relu(input=deconv2_4, output_chn=64, name='deconv3_1')
        deconv3_2 = tf.concat([deconv3_1, conv1_2], axis=4, name='deconv3_2')
        deconv3_3 = conv_bn_relu(input=deconv3_2, output_chn=32,  name='deconv3_3')
        deconv3_4 = conv_bn_relu(input=deconv3_3, output_chn=32, name='deconv3_4')
        
        pred_prob_att = conv3d(input=deconv3_4, output_chn=2, name='pred_prob_att')                   
        soft_prob_att = tf.nn.softmax(pred_prob_att, name='soft_prob_att')
        
        pred_prob_high, pred_prob_low = decom_map(soft_prob_att, deconv3_4)

        global_conv1 = global_conv3d(pred_prob_high, 32, kernel_size=13, name='gconv1')

        pred_prob_high = conv3d(input=global_conv1, output_chn=self.output_chn_syn, name='pred_prob_high')

        pred_prob_low = conv3d(input=pred_prob_low, output_chn=self.output_chn_syn, name='pred_prob_low')                  
     
        return pred_prob_high, pred_prob_low+pred_prob_high

    def fcnet_3D_model(self, inputI):
        """3D U-net"""
        
        conv1_1 = conv_bn_relu(input=inputI, output_chn=32, kernel_size=7, name='conv1_1')
        conv1_2 = conv_bn_relu(input=conv1_1, output_chn=64, kernel_size=5, name='conv1_2')
        conv1_3 = conv_bn_relu(input=conv1_2, output_chn=32, kernel_size=3, name='conv1_3')

        pred_prob_att = conv3d(input=conv1_3, output_chn=2, name='pred_prob_att')                   
        soft_prob_att = tf.nn.softmax(pred_prob_att, name='soft_prob_att')
        
        pred_prob_high, pred_prob_low = decom_map(soft_prob_att, conv1_3)

        conv1_4 = conv_bn_relu(input=pred_prob_high, output_chn=32, name='conv1_4')
        conv1_5 = conv_bn_relu(input=conv1_4, output_chn=32, name='conv1_5')
        conv1_6 = conv_bn_relu(input=conv1_5, output_chn=32, name='conv1_6')       
        
        pred_prob_high = conv3d(input=conv1_6, output_chn=self.output_chn_syn, name='pred_prob_high')

        pred_prob_low = conv3d(input=pred_prob_low, output_chn=self.output_chn_syn, name='pred_prob_low')                  
     
        return pred_prob_high, pred_prob_low+pred_prob_high

    def Inceptionnet_3D_model(self, inputI):
        """3D U-net"""
        
        conv_1 = conv_bn_relu(input=inputI, output_chn=128, kernel_size=3, name='conv_1')

        Incep_1 = Inception_layer(conv_1, 'Incep_1')
        Incep_2 = Inception_layer(Incep_1, 'Incep_2')

        pred_prob_att = conv3d(input=Incep_2, output_chn=2, name='pred_prob_att')                   
        soft_prob_att = tf.nn.softmax(pred_prob_att, name='soft_prob_att')
        
        pred_prob_high, pred_prob_low = decom_map(soft_prob_att, Incep_2)

        conv1_4 = conv_bn_relu(input=pred_prob_high, output_chn=32, name='conv1_4')
        conv1_5 = conv_bn_relu(input=conv1_4, output_chn=32, name='conv1_5')
        conv1_6 = conv_bn_relu(input=conv1_5, output_chn=32, name='conv1_6')       
        
        pred_prob_high = conv3d(input=conv1_6, output_chn=self.output_chn_syn, name='pred_prob_high')

        pred_prob_low = conv3d(input=pred_prob_low, output_chn=self.output_chn_syn, name='pred_prob_low')                  
     
        return pred_prob_high, pred_prob_low+pred_prob_high

    # train function
    def train(self):
        """Train 3D U-net"""
        
        u_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss, var_list=self.u_vars)
        
        # initialization
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
        log_file = open("./results/"+self.model_name+"_log.txt", "w")

        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS\n")
            log_file.write(" [*] Load SUCCESS\n")
        else:
            print(" [!] Load failed...\n")
            log_file.write(" [!] Load failed...\n")

        # load all volume files
        train_img, train_label,_ = load_data_pairs(self.traindata_dir)
        test_img, test_label, test_aff = load_data_pairs(self.testdata_dir)
        # temporary file to save loss
        self.test_training(0,test_img, test_label,test_aff,log_file)
        rand_idx = np.arange(len(train_img))
        start_time = time.time()
        for epoch in np.arange(self.epoch):
            np.random.shuffle(rand_idx)
            epoch_total_loss = 0.0
            for i_dx in rand_idx:
                # train batch
                batch_img, batch_label_syn, batch_label_high = get_batch_patches(train_img[i_dx], train_label[i_dx], self.inputI_size, self.batch_size, self.smooth_r)   
                feed_dict_batch={self.input_I: batch_img, 
                           self.input_gt_syn: batch_label_syn,
                           self.input_gt_high: batch_label_high
                           }
                # Update 3D U-net
                _, cur_train_loss = self.sess.run([u_optimizer, self.total_loss], feed_dict=feed_dict_batch)                
                epoch_total_loss += cur_train_loss
            
            if np.mod(epoch+1, 50) == 0:
                print("Epoch: [%2d] time: %4.4f, train_loss: %.8f\n" % (epoch+1, time.time() - start_time, epoch_total_loss/len(train_img)))
                log_file.write("Epoch: [%2d] time: %4.4f, train_loss: %.8f\n" % (epoch+1, time.time() - start_time, epoch_total_loss/len(train_img)))
                start_time = time.time()

            if epoch+1>3999 and np.mod(epoch+1, self.save_intval) == 0:
                self.save_chkpoint(self.chkpoint_dir, self.model_name, epoch+1)
                self.test_training(epoch+1,test_img, test_label,test_aff,log_file)

        log_file.close()
        
    # test the model
    def test_training(self,step,test_img, test_label, test_aff, log_file):

        # all dice
        all_mae = np.zeros([int(len(test_img))])
        all_psnr = np.zeros([int(len(test_img))])
        all_ssim = np.zeros([int(len(test_img))])
        
        for k in range(0, len(test_img)):
            
            vol_data = test_img[k]
            vol_dim = np.array(vol_data.shape).astype('int')
            
            cube_list = decompose_vol2cube(vol_data, self.inputI_size, self.inputI_chn, self.ovlp_ita)
            # predict on each cube
            cube_label_list_syn = []
            for c in range(len(cube_list)):
                cube_label_syn = self.sess.run(self.pred_prob_syn, feed_dict={self.input_I: cube_list[c]})
                cube_label_list_syn.append(cube_label_syn)

            # compose cubes into a volume
            composed_label_syn = compose_cube2vol(cube_label_list_syn, vol_dim, self.inputI_size, self.ovlp_ita, self.output_chn_syn)
            
            save_labeling_dir = self.labeling_dir+'/'+self.model_name+'/'+str(step)
            if not os.path.exists(save_labeling_dir):
                os.makedirs(save_labeling_dir)
            
            labeling_path = os.path.join(save_labeling_dir, ('%d_ct.nii.gz'%(k+36)))
            labeling_vol = nib.Nifti1Image(composed_label_syn[:,:,:,0]*255.0, test_aff[k])
            nib.save(labeling_vol, labeling_path)
            
            # evaluation                    
            k_mae, k_psnr, k_ssim = mae_psnr_ssim(composed_label_syn[:,:,:,0], test_label[k])
            print("%d: mae:%4.4f, psnr:%4.4f, ssim:%0.4f\n"%(k+1, k_mae, k_psnr, k_ssim))
            log_file.write("%d: mae:%4.4f, psnr:%4.4f, ssim:%0.4f\n"%(k+1, k_mae, k_psnr, k_ssim))
            log_file.flush()
            all_mae[k] = k_mae
            all_psnr[k] = k_psnr
            all_ssim[k] = k_ssim

        mean_mae = np.mean(all_mae, axis=0)
        mean_psnr = np.mean(all_psnr, axis=0)
        mean_ssim = np.mean(all_ssim, axis=0)
        print("mean: mae:%4.4f, psnr:%4.4f, ssim:%0.4f\n"%(mean_mae, mean_psnr, mean_ssim))
        log_file.write("mean: mae:%4.4f, psnr:%4.4f, ssim:%0.4f\n"%(mean_mae, mean_psnr, mean_ssim))
        log_file.flush()

    # save checkpoint file
    def save_chkpoint(self, checkpoint_dir, model_name, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    # load checkpoint file
    def load_chkpoint(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
    # load pretrain segmentation model
    def initialize_finetune(self):
        checkpoint_dir = './pretrain_model/sasnet_pretrain_seg_11-15-6000'
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver_ft.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
