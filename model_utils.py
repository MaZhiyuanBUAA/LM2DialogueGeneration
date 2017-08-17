#coding:utf-8

import tensorflow as tf
from model import LM,Seq2seqEmbeddedLM,Seq2seqEmbeddedLM_consist
from config import lm_config,dm_config,dmc_config
lm_config = lm_config()
dm_config = dm_config()
dmc_config = dmc_config()

def create_LM(session):
  model = LM(lm_config.vocab_size,
             lm_config.emb_dim,
             lm_config.size,
             lm_config.num_steps,
             lm_config.num_layers,
             lm_config.learning_rate,
             lm_config.decay_factor,
             lm_config.batch_size,
             lm_config.max_gradient_norm)
  ckpt = tf.train.get_checkpoint_state(lm_config.model_dir)
  if ckpt:
    print("Reading language model from %s "%ckpt.model_checkpoint_path)
    model.saver.restore(session,ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh params.")
    session.run(tf.global_variables_initializer())
  return model

def create_dialogueModel(session):
  lm_model = create_LM(session)
  model = Seq2seqEmbeddedLM(lm_model,
                            dm_config.input_len,
                            dm_config.output_len,
                            dm_config.size,
                            dm_config.num_layers,
                            dm_config.learning_rate,
                            dm_config.decay_factor,
                            dm_config.batch_size,
                            dm_config.max_gradient_norm,
                            dm_config.new_embeddings,
                            dm_config.vocab_size,
                            dm_config.emb_dim,
                            dm_config.num_heads,
                            name='seq2seq_lm')
  ckpt = tf.train.get_checkpoint_state(dm_config.model_dir)
  if ckpt:
    print("Reading seq2seq Embedded LM from %s"%ckpt.model_checkpoint_path)
    model.saver.restore(session,ckpt.model_checkpoint_path)
    session.run(model.assign_params)
    session.run(tf.assign(model.learning_rate,dm_config.learning_rate))
  else:
    print("Created model with fresh params.")
    session.run(tf.global_variables_initializer())
    #session.run(model.assign_params)
  return model

def create_dialogueModel_consist(session):
  lm_model = create_LM(session)
  model = Seq2seqEmbeddedLM_consist(lm_model,
                            dmc_config.input_len,
                            dmc_config.output_len,
                            dmc_config.size,
                            dmc_config.num_layers,
                            dmc_config.learning_rate,
                            dmc_config.decay_factor,
                            dmc_config.batch_size,
                            dmc_config.max_gradient_norm,
                            dmc_config.vocab_size,
                            dmc_config.emb_dim,
                            dmc_config.num_heads,
                            name='seq2seq_lm_consist')
  ckpt = tf.train.get_checkpoint_state(dmc_config.model_dir)
  if ckpt:
    print("Reading seq2seq Embedded LM from %s"%ckpt.model_checkpoint_path)
    model.saver.restore(session,ckpt.model_checkpoint_path)
    session.run(model.assign_params)
    session.run(tf.assign(model.learning_rate,dmc_config.learning_rate))
  else:
    print("Created model with fresh params.")
    session.run(tf.global_variables_initializer())
    #session.run(model.assign_params)
  return model

if __name__ == '__main__':
  with tf.Session() as sess:
    model=create_dialogueModel(sess)
