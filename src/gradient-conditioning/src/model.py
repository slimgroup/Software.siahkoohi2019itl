import os
import time
import tensorflow as tf
import numpy as np
from collections import namedtuple
import h5py
from module import *
from utils import *
from math import floor
from random import shuffle
import random

class gradient_conditioning(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size0 = args.image_size0
        self.image_size1 = args.image_size1
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.experiment_dir = args.experiment_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.transfer = args.transfer

        self.file_name_trainA = os.path.join(args.data_path, 'gradientCorrection_A_freq40-wide_picked_train.hdf5')
        self.file_name_trainB = os.path.join(args.data_path, 'gradientCorrection_B_freq40-wide_picked_train.hdf5')
        self.file_name_testA  = os.path.join(args.data_path, 'gradientCorrection_A_freq40-wide_test.hdf5')
        self.file_name_testB  = os.path.join(args.data_path, 'gradientCorrection_B_freq40-wide_test.hdf5')

        if self.transfer==1:
            self.file_name_trainA = self.file_name_testA
            self.file_name_trainB = self.file_name_testB

        self.dataset_name_train  = "train_dataset"
        self.dataset_name_test   = "test_dataset"

        if self.transfer==1:
            self.dataset_name_train = self.dataset_name_test

        self.file_trainA = h5py.File(self.file_name_trainA, 'r')
        self.file_trainB = h5py.File(self.file_name_trainB, 'r')
        self.file_testA  = h5py.File(self.file_name_testA, 'r')
        self.file_testB  = h5py.File(self.file_name_testB, 'r')

        self.data_num  = self.file_trainA[self.dataset_name_train].shape[0]
        self.test_num  = self.file_testB[self.dataset_name_test].shape[0]

        self.discriminator = discriminator
        self.generator = generator_resnet
        self.criterionGAN = mae_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size0 image_size1 \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.image_size0, args.image_size1,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver()

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size0, self.image_size1,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        self.equality_factor  = tf.placeholder(tf.float32, None, name='equality_factor')
        self.SNR_diff = tf.placeholder(tf.float32, [None, self.image_size0*self.image_size1], name='SNR_diff')
        self.SNR_real = tf.placeholder(tf.float32, [None, self.image_size0*self.image_size1], name='SNR_real')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_A = self.generator(self.real_B, self.options, False, name="generatorB2A")

        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake))

        self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.equality_factor * abs_criterion(self.fake_A, self.real_A)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size0, self.image_size1,
                                             self.input_c_dim], name='fake_A_sample')

        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss

        self.Rec_SNR = -20.0* tf.log(tf.norm(self.SNR_diff, ord='euclidean')/tf.norm(self.SNR_real, ord='euclidean'))/tf.log(10.0)

        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_b2a_sum, self.g_loss_sum])
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.d_loss_sum]
        )
        self.Rec_SNR_train_sum = tf.summary.scalar("Rec_SNR_train", self.Rec_SNR)
        self.Rec_SNR_test_sum  = tf.summary.scalar("Rec_SNR_test", self.Rec_SNR)

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size0, self.image_size1,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size0, self.image_size1,
                                      self.output_c_dim], name='test_B')
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()

        if self.transfer==0:
            self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
            self.g_vars = [var for var in t_vars if 'generator' in var.name]
        elif self.transfer==1:
            self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
            self.g_vars = [var for var in t_vars if 'generator' in var.name]
            self.d_vars  = self.d_vars[-4:]
            self.g_vars  = self.g_vars[9:]

        for var in self.d_vars: print(var.name)
        for var in self.g_vars: print(var.name)        

        var_size = 0
        for var in self.d_vars:
            var_size = var_size + int(np.prod(np.array(var.shape)))
        for var in self.g_vars:
            var_size = var_size + int(np.prod(np.array(var.shape)))
        print(("Number of unknowns: %d" % (var_size)))

    def train(self, args):

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if self.transfer==0:
            batch_idxs = list(range(int(floor(float(self.data_num) / self.batch_size))))
        elif self.transfer==1:
            batch_idxs = list(range(0, int(floor(float(self.data_num) / self.batch_size)), 20))

        for epoch in range(args.epoch):

            shuffle(batch_idxs)
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

            for idx in range(0, len(batch_idxs)):

                batch_images = load_train_data(batch_idxs[idx], batch_size=self.batch_size, \
                    fileA=self.file_trainA, fileB=self.file_trainB, dataset=self.dataset_name_train)
                batch_images = np.array(batch_images).astype(np.float32)

                # Update G network and record fake outputs
                fake_A, _, summary_str = self.sess.run(
                    [self.fake_A, self.g_optim, self.g_sum],
                    feed_dict={self.real_data: batch_images, self.lr: lr, self.equality_factor: self.L1_lambda})
                self.writer.add_summary(summary_str, counter)

                # Update D network
                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],
                    feed_dict={self.real_data: batch_images,
                               self.fake_A_sample: fake_A,
                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (
                    epoch, int(idx), int(len(batch_idxs)), time.time() - start_time)))

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx, counter-1)

                if np.mod(counter, int(floor(args.save_freq/self.batch_size))) == 2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.experiment_dir, self.image_size0)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.experiment_dir, self.image_size0)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx, counter):

        if self.transfer==0:
            res_rnd = int(np.random.randint(0, self.data_num))
        elif self.transfer==1:
                    res_rnd = int(random.choice(list(range(0, \
                        int(floor(float(self.data_num) / self.batch_size)), 20))))

        train_images = load_train_data(res_rnd, is_testing=True, batch_size=self.batch_size, \
            fileA=self.file_trainA, fileB=self.file_trainB, dataset=self.dataset_name_train)
        train_images = np.array(train_images).astype(np.float32)

        sample_A = train_images[:, :, :, :self.input_c_dim]
        sample_B = train_images[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        out_var, in_var = (self.testA, self.test_B)

        fake_A = self.sess.run(out_var, feed_dict={in_var: sample_B})

        result_img = fake_A
        diff_img = np.absolute(sample_A-result_img)

        diff_img_real = diff_img[:, :, :, 0].reshape((1, self.image_size0*self.image_size1))
        sample_A_real = sample_A[:, :, :, 0].reshape((1, self.image_size0*self.image_size1))

        Rec_SNR, summary_str = self.sess.run(
            [self.Rec_SNR, self.Rec_SNR_train_sum],
            feed_dict={self.SNR_diff: diff_img_real,
                       self.SNR_real: sample_A_real})
        self.writer.add_summary(summary_str, counter)

        print(("Recovery SNR (training data): %4.4f" % (Rec_SNR)))


    def test(self, args):

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        out_var, in_var = (self.testA, self.test_B)

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

####################################

        SNR_AVG0 = 0
        SNR_AVG1 = 0
        iii = 0

        strResult = os.path.join(self.sample_dir, 'mapping_SNR.hdf5')

        if os.path.isfile(strResult):
            os.remove(strResult)

        file_SNR = h5py.File(strResult, 'w-')
        dataset_str = "SNR"
        datasetSNR = file_SNR.create_dataset(dataset_str, (1, 1))

        strCorrection = os.path.join(self.sample_dir, 'mapping_result.hdf5')

        if os.path.isfile(strCorrection):
            os.remove(strCorrection)

        file_correction = h5py.File(strCorrection, 'w-')
        datasetCorrection_str = "result"
        datasetCorrection = file_correction.create_dataset(datasetCorrection_str, (self.test_num, self.image_size0, self.image_size1))        
        datasetCorrectionA = file_correction.create_dataset(datasetCorrection_str + 'A', (self.test_num, self.image_size0, self.image_size1))
        datasetCorrectionB = file_correction.create_dataset(datasetCorrection_str + 'B', (self.test_num, self.image_size0, self.image_size1))

        start_time_interp = time.time()

        for itest in range(0, self.test_num):
            print(itest)
            res_rnd = itest
            # res_rnd =
            sample_B = load_test_data(res_rnd, filetest=self.file_testB, dataset=self.dataset_name_test)
            sample_B = np.array(sample_B).astype(np.float32)

            sample_A = load_test_data(res_rnd, filetest=self.file_testA, dataset=self.dataset_name_test)
            sample_A = np.array(sample_A).astype(np.float32)

            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_B})
            datasetCorrection[itest, :, :] = fake_img[0, :, :, 0]
            datasetCorrectionB[itest, :, :] = sample_B[0, :, :, 0]
            datasetCorrectionA[itest, :, :] = sample_A[0, :, :, 0]

            result_img = fake_img
            diff_img = np.absolute(sample_A-result_img)

            diff_img_real = diff_img[:, :, :, 0].reshape((1, self.image_size0*self.image_size1))
            sample_A_real = sample_A[:, :, :, 0].reshape((1, self.image_size0*self.image_size1))

            Rec_SNR_real = self.sess.run(
                [self.Rec_SNR],
                feed_dict={self.SNR_diff: diff_img_real,
                           self.SNR_real: sample_A_real})

            print(("Recovery SNR for gradient # itest: %4.4f" % (Rec_SNR_real[0])))

            SNR_AVG0 = SNR_AVG0 + Rec_SNR_real[0]
            iii = iii + 1

        datasetSNR[0, 0] = SNR_AVG0/iii
        print(datasetSNR[0, 0])
        file_SNR.close()
        file_correction.close()