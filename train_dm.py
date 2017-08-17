#coding:utf-8

import tensorflow as tf
from model_utils import create_LM,create_dialogueModel,create_dialogueModel_consist
from config import lm_config,dm_config,dmc_config
import data_utils
from data_utils import read_data_lm,read_data
import time,os
import math
import sys

lm_config = lm_config()
dm_config = dm_config()
dmc_config = dmc_config()
def train_LM():
    print("Preparing dialog data in %s" % lm_config.data_dir)
    train_data, dev_data, _ = data_utils.prepare_dialog_data(lm_config.data_dir, lm_config.vocab_size)

    with tf.Session() as sess:

        # Create model.
        print("Creating %d layers of %d units." % (lm_config.num_layers, lm_config.size))
        #model = create_LM(sess)
        model = create_LM(sess)

        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)." % lm_config.max_train_data_size)
        dev_set = read_data_lm(dev_data)
        train_set = read_data_lm(train_data, lm_config.max_train_data_size)
        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        max_global_step = 120000
        while model.global_step.eval()<max_global_step:
          # Get a batch and make a step.
          start_time = time.time()
          inputs,weights = model.get_batch(train_set)
          time_ = time.time()-start_time
          #print(time_)
          step_loss,_ = model.step(sess, inputs,weights)
          #print(time.time()-start_time-time_)
          step_time += (time.time() - start_time) / lm_config.steps_per_checkpoint
          loss += step_loss / lm_config.steps_per_checkpoint
          current_step += 1
          #print('loss:%f'%loss)
          # Once in a while, we save checkpoint, print statistics, and run evals.
          if current_step % lm_config.steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            perplexity = math.exp(loss) if loss < 300 else float('inf')
            print ("global step %d learning rate %.4f step-time %.2f perplexity %.2f" %
                   (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))

            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
              sess.run(model.learning_rate_decay_op)

            previous_losses.append(loss)

            # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(lm_config.model_dir, "model.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            step_time, loss = 0.0,0.0
            sys.stdout.flush()
def train_dm():
    print("Preparing dialog data in %s" % dm_config.data_dir)
    train_data, dev_data, _ = data_utils.prepare_dialog_data(dm_config.data_dir, dm_config.vocab_size)

    with open(dm_config.model_dir+'/results.txt','a') as f:
        sess = tf.Session()
        # Create model.
        print("Creating %d layers of %d units." % (dm_config.num_layers, dm_config.size))
        #model = create_LM(sess)
        model = create_dialogueModel(sess)

        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)." % dm_config.max_train_data_size)
        dev_set = read_data(dev_data)
        train_set = read_data(train_data,dm_config.max_train_data_size)
        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        max_global_step = 120000
        while model.global_step.eval(session=sess)<max_global_step:
          # Get a batch and make a step.
          start_time = time.time()
          inputs,targs,weights = model.get_batch(train_set)
          time_ = time.time()-start_time
          #print(time_)
          step_loss,_ = model.step(sess, inputs,targs,weights)
          #print(time.time()-start_time-time_)
          step_time += (time.time() - start_time) / dm_config.steps_per_checkpoint
          loss += step_loss / dm_config.steps_per_checkpoint
          current_step += 1
          #print('loss:%f'%loss)
          # Once in a while, we save checkpoint, print statistics, and run evals.
          if current_step%30000==0 and current_step>0:
            sess.close()
            sess = tf.Session()
            model = create_dialogueModel(sess)
          if current_step % dm_config.steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            perplexity = math.exp(loss) if loss < 300 else float('inf')
            f.write("global step %d learning rate %.4f step-time %.2f perplexity %.2f\n" %
                   (model.global_step.eval(session=sess), model.learning_rate.eval(session=sess), step_time, perplexity))

            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
              sess.run(model.learning_rate_decay_op)

            previous_losses.append(loss)

            # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(dm_config.model_dir, "model.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            step_time, loss = 0.0,0.0
            #sys.stdout.flush()
def train_dmc():
    print("Preparing dialog data in %s" % dmc_config.data_dir)
    train_data, dev_data, _ = data_utils.prepare_dialog_data(dmc_config.data_dir, dmc_config.vocab_size)

    with tf.Session() as sess:

        # Create model.
        print("Creating %d layers of %d units." % (dmc_config.num_layers, dmc_config.size))
        #model = create_LM(sess)
        model = create_dialogueModel_consist(sess)

        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)." % dmc_config.max_train_data_size)
        dev_set = read_data(dev_data)
        train_set = read_data(train_data,dmc_config.max_train_data_size)
        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        max_global_step = 120000
        while model.global_step.eval()<max_global_step:
          # Get a batch and make a step.
          start_time = time.time()
          inputs,targs,weights = model.get_batch(train_set)
          time_ = time.time()-start_time
          #print(time_)
          step_loss,_ = model.step(sess, inputs,targs,weights)
          #print(time.time()-start_time-time_)
          step_time += (time.time() - start_time) / dmc_config.steps_per_checkpoint
          loss += step_loss / dmc_config.steps_per_checkpoint
          current_step += 1
          #print('loss:%f'%loss)
          # Once in a while, we save checkpoint, print statistics, and run evals.
          if current_step % dmc_config.steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            perplexity = math.exp(loss) if loss < 300 else float('inf')
            print ("global step %d learning rate %.4f step-time %.2f perplexity %.2f" %
                   (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))

            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
              sess.run(model.learning_rate_decay_op)

            previous_losses.append(loss)

            # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(dmc_config.model_dir, "model.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            step_time, loss = 0.0,0.0
            sys.stdout.flush()

if __name__ == '__main__':
  train_dm()  
  #train_dmc()

