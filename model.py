#coding:utf-8
import copy
import tensorflow as tf
import random
from seq2seq import sequence_loss
import data_utils
import numpy as np
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

linear = core_rnn_cell_impl._linear

class LM(object):
  def __init__(self,vocab_size,emb_dim,size,num_steps,num_layers,
               learning_rate,decay_factor,batch_size,max_gradient_norm,
               num_samples=512,name='LM'):
    with tf.variable_scope(name) as vs:
      self.vocab_size = vocab_size
      self.num_steps = num_steps
      self.batch_size = batch_size
      self.learning_rate = tf.Variable(float(learning_rate),trainable=False)
      self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate*decay_factor)
      self.global_step = tf.Variable(0,trainable=False)
      self.inputs = [tf.placeholder(tf.int32,
                                  shape=[None],
                                  name='input{}'.format(i)) for i in range(num_steps)]
      self.targets = self.inputs[1:]+[tf.placeholder(tf.int32,
                                                   shape=[None],
                                                   name='last_target')]
      self.target_weights = [tf.placeholder(tf.float32,
                                          shape=[None],
                                          name='weight{}'.format(i)) for i in range(num_steps)]
      self.embeddings = tf.get_variable(name='embeddings',
                                      shape=[vocab_size,emb_dim],
                                      initializer=tf.random_uniform_initializer())
      emb_inputs = [tf.nn.embedding_lookup(self.embeddings,ele) for ele in self.inputs]
      self.cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(size) for i in range(num_layers)])
      self.outputs,_ = tf.contrib.rnn.static_rnn(self.cell,emb_inputs,dtype=tf.float32)

      self.proj_w = tf.get_variable(name='proj_w',
                             shape=[size,vocab_size],
                             initializer=tf.random_uniform_initializer())
      self.proj_b = tf.get_variable(name='proj_b',
                             shape=[vocab_size],
                             initializer=tf.random_uniform_initializer())
      self.w_t = tf.transpose(self.proj_w)

      def nce_loss(labels,inputs):
        labels = tf.reshape(labels,[-1,1])
        return tf.nn.sampled_softmax_loss(self.w_t,self.proj_b,labels,inputs,num_samples,self.vocab_size)
      #self.losses = tf.reduce_mean(nce_loss(tf.concat(self.targets,axis=0),tf.concat(outputs,axis=0)))
      self.losses = sequence_loss(self.outputs,self.targets,self.target_weights,softmax_loss_function=nce_loss)
    self.predict = [tf.nn.xw_plus_b(ele,self.proj_w,self.proj_b) for ele in self.outputs]

    self.params = []
    for ele in tf.global_variables():
      if ele.name.startswith(name):
        self.params.append(ele)
    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
    gradients = tf.gradients(self.losses,self.params)
    clipped_gradients,norm = tf.clip_by_global_norm(gradients,max_gradient_norm)
    self.update = opt.apply_gradients(zip(clipped_gradients,self.params),global_step=self.global_step)
    self.saver = tf.train.Saver(self.params)
    tmp = [ele.name for ele in self.params]
    print(len(tmp))
    print('\n'.join(tmp))


  def step(self,sess,inps,weights,forward_only=False):
    if len(inps)!=self.num_steps:
      raise ValueError('input length(%d) must be equal to the num_steps(%d)'%(len(inputs),self.num_steps))
    if len(weights)!=self.num_steps:
      raise ValueError('weight length(%d) must be equal to the num_steps(%d)'%(len(weights),self.num_steps))
    
    input_feed = {}
    for i in range(self.num_steps):
      input_feed[self.inputs[i].name] = inps[i]
      input_feed[self.target_weights[i].name] = weights[i]
    input_feed[self.targets[-1].name] = np.zeros([self.batch_size],dtype=np.int32)
    
    if not forward_only:
       output_feed = [self.losses,self.update]
    else:
       output_feed = [self.losses,self.predict]
    outputs = sess.run(output_feed,input_feed)
    return outputs[0],outputs[1]

  def get_batch(self, data):
    inputs = []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in range(self.batch_size):

      inp = random.choice(data)
      inp_pad = [data_utils.PAD_ID] * (self.num_steps - len(inp))
      inputs.append(list(inp + inp_pad))
    
    batch_inputs, batch_weights = [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in range(self.num_steps):
      batch_inputs.append(
          np.array([inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in range(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < self.num_steps - 1:
          target = inputs[batch_idx][length_idx + 1]
        if length_idx == self.num_steps - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_inputs, batch_weights

class Seq2seqEmbeddedLM(object):
  def __init__(self,LM,input_len,output_len,size,num_layers,
               learning_rate,decay_factor,batch_size,max_gradient_norm,
               new_embeddings=False,vocab_size=None,emb_dim=None,
               num_heads=1,num_samples=512,name='Seq2seqEmbddedLM'):
    with tf.variable_scope(name) as vs:
      self.input_len = input_len
      self.output_len = output_len
      self.batch_size = batch_size
      self.learning_rate = tf.Variable(float(learning_rate),trainable=False)
      self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate*decay_factor)
      self.global_step = tf.Variable(0,trainable=False)
      self.inputs = [tf.placeholder(tf.int32,shape=[None],name='input{}'.format(i)) for i in range(input_len)]
      self.decoder_inputs = [tf.placeholder(tf.int32,shape=[None],name='decoder_input{}'.format(i)) for i in range(output_len)]
      self.targets = self.decoder_inputs[1:]+[tf.placeholder(tf.int32,shape=[None],name='last_target')]
      self.target_weights = [tf.placeholder(tf.float32,shape=[None],name='target_weight{}'.format(i)) for i in range(output_len)]

      with tf.variable_scope('use_LM'):
        embeddings = tf.get_variable(name='embeddings',shape=[vocab_size,emb_dim],dtype=tf.float32,initializer=tf.random_uniform_initializer())
     
      emb_inps = [tf.nn.embedding_lookup(embeddings,ele) for ele in self.inputs]
      emb_decoder_inps = [tf.nn.embedding_lookup(embeddings,ele) for ele in self.decoder_inputs]
 
      cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(size) for i in range(num_layers)])
      encoder_outputs,_ = tf.contrib.rnn.static_rnn(cell,emb_inps,dtype=tf.float32)
     
      top_states = [tf.reshape(ele,[-1,1,cell.output_size]) for ele in encoder_outputs]
      att_states = tf.concat(top_states,axis=1)
      att_length = att_states.get_shape()[1].value
      att_size = att_states.get_shape()[2].value
      att_vec_size = att_size
  
      hidden = tf.reshape(att_states,[-1,att_length,1,att_size])
      hidden_features = []
      v = []
      for a in range(num_heads):
        k = tf.get_variable('AttnW_%d'%a,[1,1,att_size,att_vec_size])
        hidden_features.append(tf.nn.conv2d(hidden,k,[1,1,1,1],'SAME'))
        v.append(tf.get_variable('AttnV_%d'%a,[att_vec_size]))
      
      def attention(query):
        """Put attention masks on hidden using hidden_features and query."""
        ds = []  # Results of attention reads will be stored here.
        #if tf.contrib.framework.nest.is_sequence(query):  # If the query is a tuple, flatten it.
        #  query_list = tf.contrib.framework.nest.flatten(query)
        #  for q in query_list:  # Check that ndims == 2 if specified.
        #    ndims = q.get_shape().ndims
        #    if ndims:
        #      assert ndims == 2
        #  query = tf.concat(query_list, 1)
        for a in xrange(num_heads):
          with tf.variable_scope("Attention_%d" % a):
            y = linear(query, att_vec_size, True)
            y = tf.reshape(y, [-1, 1, 1, att_vec_size])
            # Attention mask is a softmax of v^T * tanh(...).
            s = tf.reduce_sum(v[a] * tf.tanh(hidden_features[a] + y),
                                  [2, 3])
            a_ = tf.nn.softmax(s)
            # Now calculate the attention-weighted vector d.
            d = tf.reduce_sum(
                tf.reshape(a_, [-1, att_length, 1, 1]) * hidden, [1, 2])
            ds.append(tf.reshape(d, [-1, att_size]))
        return ds
      #decoder_cell = LM.cell 
      with tf.variable_scope('loop'):
        decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(size) for i in range(num_layers)])
        state = decoder_cell.zero_state(LM.batch_size,dtype=tf.float32)
        #print(state)
        #state = tf.zeros([self.batch_size,size])
        attns = attention(state)
        #with tf.variable_scope('first_inp'):
        #  inp = linear(attns,emb_dim,True)
        outputs = []
        for i,inp in enumerate(emb_decoder_inps):
          if i>0:
            variable_scope.get_variable_scope().reuse_variables()
          x = linear([inp]+attns,emb_dim,True)
          with tf.variable_scope('use_LM'):
            cell_output,state = decoder_cell(x,state)
          with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                             reuse=True):
            attns = attention(state)
          with tf.variable_scope('AttnOutputProjection'):
            output = linear([cell_output]+attns,emb_dim,True)
          outputs.append(output)
      with tf.variable_scope('use_LM'):
        proj_w = tf.get_variable('proj_w',[emb_dim,vocab_size])
        proj_b = tf.get_variable('proj_b',[vocab_size])
        w_t = tf.transpose(proj_w)
      def nce_loss(labels,inputs):
        labels = tf.reshape(labels,[-1,1])
        return tf.nn.sampled_softmax_loss(w_t,proj_b,labels,inputs,num_samples,vocab_size)
      #self.losses = tf.reduce_mean(nce_loss(tf.concat(self.targets,axis=0),tf.concat(outputs,axis=0)))
      self.losses = sequence_loss(outputs,self.targets,self.target_weights,softmax_loss_function=nce_loss)
    self.predict = [tf.nn.xw_plus_b(ele,LM.proj_w,LM.proj_b) for ele in outputs]
    self.params = []
    self.params_use_LM = []
    for ele in tf.global_variables():
      if ele.name.startswith(name):
        if ele.name.find('use_LM')>-1:
          self.params_use_LM.append(ele)
        else:
          self.params.append(ele)
    #assign params of LM
    self.assign_params = [tf.assign(ele,LM.params[ind+2]) for ind, ele in enumerate(self.params_use_LM)]
    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
    gradients = tf.gradients(self.losses,self.params+self.params_use_LM)
    clipped_gradients,norm = tf.clip_by_global_norm(gradients,max_gradient_norm)
    self.update = opt.apply_gradients(zip(clipped_gradients,self.params+self.params_use_LM),global_step=self.global_step)
    self.saver = tf.train.Saver(self.params+self.params_use_LM)
    tmp = [ele.name for ele in self.params]
    print(len(tmp))
    print('\n'.join(tmp))
    tmp = [ele.name for ele in self.params_use_LM]
    print(len(tmp))
    print('\n'.join(tmp))
   
  def step(self,sess,encoder_inps,decoder_inps,weights,forward_only=False):
    if len(encoder_inps)!=self.input_len:
      raise ValueError('input length(%d) must be equal to the input_len(%d)'%(len(inputs),self.input_len))
    if len(weights)!=self.output_len:
      raise ValueError('weight length(%d) must be equal to the output_len(%d)'%(len(weights),self.output_len))
    if len(decoder_inps)!= self.output_len:
      raise ValueError('target length(%d) must be equal to the output_len(%d)'%(len(weights),self.output_len))
    input_feed = {}
    for i in range(self.input_len):
      input_feed[self.inputs[i].name] = encoder_inps[i]
    for i in range(self.output_len):
      input_feed[self.target_weights[i].name] = weights[i]
      input_feed[self.decoder_inputs[i].name] = decoder_inps[i]
    input_feed[self.targets[-1].name] = np.zeros(self.batch_size)
    #input_feed[self.targets[-1].name] = np.zeros([self.batch_size],dtype=np.int32)

    if not forward_only:
       output_feed = [self.losses,self.update]
    else:
       output_feed = [self.losses,self.predict]
    outputs = sess.run(output_feed,input_feed)
    return outputs[0],outputs[1]

  def get_batch(self, data):
    encoder_inputs, decoder_inputs = [], []
    for _ in range(self.batch_size):

        encoder_input, decoder_input = random.choice(data)
        # Encoder inputs are padded and then reversed.
        encoder_pad = [data_utils.PAD_ID] * (self.input_len - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = self.output_len - len(decoder_input)
        decoder_inputs.append(decoder_input +
                              [data_utils.PAD_ID] * decoder_pad_size)

    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in range(self.input_len):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))
    for length_idx in range(self.output_len):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))
      
      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in range(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx<self.output_len-1:
          target = decoder_inputs[batch_idx][length_idx+1]

        if length_idx==self.output_len-1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs,batch_weights

class Seq2seqEmbeddedLM_consist(object):
  def __init__(self,LM,input_len,output_len,size,num_layers,
               learning_rate,decay_factor,batch_size,max_gradient_norm,
               vocab_size=None,emb_dim=None,
               num_heads=1,num_samples=512,name='Seq2seqEmbddedLM'):
    with tf.variable_scope(name) as vs:
      self.input_len = input_len
      self.output_len = output_len
      self.batch_size = batch_size
      self.learning_rate = tf.Variable(float(learning_rate),trainable=False)
      self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate*decay_factor)
      self.global_step = tf.Variable(0,trainable=False)
      self.inputs = [tf.placeholder(tf.int32,shape=[None],name='input{}'.format(i)) for i in range(input_len)]
      self.targets = [tf.placeholder(tf.int32,shape=[None],name='target{}'.format(i)) for i in range(output_len)]
      #self.targets = self.decoder_inputs[1:]+[tf.placeholder(tf.int32,shape=[None],name='last_target')]
      self.target_weights = [tf.placeholder(tf.float32,shape=[None],name='target_weight{}'.format(i)) for i in range(output_len)]

      with tf.variable_scope('use_LM'):
        embeddings = tf.get_variable('embeddings',[vocab_size,emb_dim],tf.float32,initializer=tf.random_uniform_initializer())

      emb_inps = [tf.nn.embedding_lookup(embeddings,ele) for ele in self.inputs]
      #emb_decoder_inps = [tf.nn.embedding_lookup(embeddings,ele) for ele in self.decoder_inputs]

      cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(size) for i in range(num_layers)])
      encoder_outputs,_ = tf.contrib.rnn.static_rnn(cell,emb_inps,dtype=tf.float32)

      top_states = [tf.reshape(ele,[-1,1,cell.output_size]) for ele in encoder_outputs]
      att_states = tf.concat(top_states,axis=1)
      att_length = att_states.get_shape()[1].value
      att_size = att_states.get_shape()[2].value
      att_vec_size = att_size
      hidden = tf.reshape(att_states,[-1,att_length,1,att_size])
      hidden_features = []
      v = []
      for a in range(num_heads):
        k = tf.get_variable('AttnW_%d'%a,[1,1,att_size,att_vec_size])
        hidden_features.append(tf.nn.conv2d(hidden,k,[1,1,1,1],'SAME'))
        v.append(tf.get_variable('AttnV_%d'%a,[att_vec_size]))

      def attention(query):
        """Put attention masks on hidden using hidden_features and query."""
        ds = []  # Results of attention reads will be stored here.
        #if tf.contrib.framework.nest.is_sequence(query):  # If the query is a tuple, flatten it.
        #  query_list = tf.contrib.framework.nest.flatten(query)
        #  for q in query_list:  # Check that ndims == 2 if specified.
        #    ndims = q.get_shape().ndims
        #    if ndims:
        #      assert ndims == 2
        #  query = tf.concat(query_list, 1)
        for a in xrange(num_heads):
          with tf.variable_scope("Attention_%d" % a):
            y = linear(query, att_vec_size, True)
            y = tf.reshape(y, [-1, 1, 1, att_vec_size])
            # Attention mask is a softmax of v^T * tanh(...).
            s = tf.reduce_sum(v[a] * tf.tanh(hidden_features[a] + y),
                                  [2, 3])
            a_ = tf.nn.softmax(s)
            # Now calculate the attention-weighted vector d.
            d = tf.reduce_sum(
                tf.reshape(a_, [-1, att_length, 1, 1]) * hidden, [1, 2])
            ds.append(tf.reshape(d, [-1, att_size]))
        return ds
      with tf.variable_scope('loop'):
        #decoder_cell = LM.cell
        decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(size) for i in range(num_layers)])
        state = decoder_cell.zero_state(self.batch_size,dtype=tf.float32)
        #print(state)
        #state = tf.zeros([self.batch_size,size])
        attns = attention(state)
        with tf.variable_scope('first_inp'):
          inp = linear(attns,emb_dim,True)
        outputs = []
     
        for i in range(self.output_len):
          if i>0:
            variable_scope.get_variable_scope().reuse_variables()
          x = linear([inp]+attns,emb_dim,True)
          with tf.variable_scope('use_LM'):
            cell_output,state = decoder_cell(x,state)
          with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                             reuse=True):
            attns = attention(state)
          with tf.variable_scope('AttnOutputProjection'):
            inp = linear([cell_output]+attns,emb_dim,True)
          outputs.append(inp)
      with tf.variable_scope('use_LM'):
        proj_w = tf.get_variable('proj_w',[emb_dim,vocab_size],dtype=tf.float32,initializer=tf.random_uniform_initializer())
        proj_b = tf.get_variable('proj_b',[vocab_size],dtype=tf.float32,initializer=tf.random_uniform_initializer())
      w_t = tf.transpose(proj_w)
      #self.assign_proj = [tf.assign(proj_w,LM.proj_w),tf.assign(proj_b,LM.proj_b),tf.assign(w_t,LM.w_t)]
      def nce_loss(labels,inputs):
        labels = tf.reshape(labels,[-1,1])
        return tf.nn.sampled_softmax_loss(w_t,proj_b,labels,inputs,num_samples,vocab_size)
      #self.losses = tf.reduce_mean(nce_loss(tf.concat(self.targets,axis=0),tf.concat(outputs,axis=0)))
      self.losses = sequence_loss(outputs,self.targets,self.target_weights,softmax_loss_function=nce_loss)
    self.predict = [tf.nn.xw_plus_b(ele,proj_w,proj_b) for ele in outputs]
    self.params = []
    self.params_use_LM = []
    for ele in tf.global_variables():
      if ele.name.startswith(name):
        if ele.name.find('use_LM')>-1:
          self.params_use_LM.append(ele)
        else:
          self.params.append(ele)
    #assign params of LM
    self.assign_params = [tf.assign(ele,LM.params[ind+2]) for ind, ele in enumerate(self.params_use_LM)]
    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
    gradients = tf.gradients(self.losses,self.params+self.params_use_LM)
    clipped_gradients,norm = tf.clip_by_global_norm(gradients,max_gradient_norm)
    self.update = opt.apply_gradients(zip(clipped_gradients,self.params+self.params_use_LM),global_step=self.global_step)
    self.saver = tf.train.Saver(self.params+self.params_use_LM)
    tmp = [ele.name for ele in self.params]
    print(len(tmp))
    print('\n'.join(tmp))
    tmp = [ele.name for ele in self.params_use_LM]
    print(len(tmp))
    print('\n'.join(tmp))

  def step(self,sess,inps,targs,weights,forward_only=False):
    if len(inps)!=self.input_len:
      raise ValueError('input length(%d) must be equal to the input_len(%d)'%(len(inputs),self.input_len))
    if len(weights)!=self.output_len:
      raise ValueError('weight length(%d) must be equal to the output_len(%d)'%(len(weights),self.output_len))
    if len(targs)!= self.output_len:
      raise ValueError('target length(%d) must be equal to the output_len(%d)'%(len(targs),self.output_len))
    input_feed = {}
    for i in range(self.input_len):
      input_feed[self.inputs[i].name] = inps[i]
    for i in range(self.output_len):
      input_feed[self.target_weights[i].name] = weights[i]
      input_feed[self.targets[i].name] = targs[i]
    #input_feed[self.targets[-1].name] = np.zeros(self.batch_size)
    #input_feed[self.targets[-1].name] = np.zeros([self.batch_size],dtype=np.int32)

    if not forward_only:
       output_feed = [self.losses,self.update]
    else:
       output_feed = [self.losses,self.predict]
    outputs = sess.run(output_feed,input_feed)
    return outputs[0],outputs[1]

  def get_batch(self, data):
    encoder_inputs, decoder_outputs = [], []
    for _ in range(self.batch_size):

        encoder_input, decoder_output = random.choice(data)
        # Encoder inputs are padded and then reversed.
        encoder_pad = [data_utils.PAD_ID] * (self.input_len - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = self.output_len - len(decoder_output)
        decoder_outputs.append(decoder_output +
                              [data_utils.PAD_ID] * decoder_pad_size)

    batch_encoder_inputs, batch_decoder_outputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in range(self.input_len):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))
    for length_idx in range(self.output_len):
      batch_decoder_outputs.append(
          np.array([decoder_outputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in range(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward
        target = decoder_outputs[batch_idx][length_idx]

        if  target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_outputs,batch_weights

class cnn2seq(object):
  def __init__(self,LM,input_len,output_len,size,num_layers,
               learning_rate,decay_factor,batch_size,max_gradient_norm,
               new_embeddings=False,vocab_size=None,emb_dim=None,
               num_heads=1,num_samples=512,name='Seq2seqEmbddedLM'):
    with tf.variable_scope(name) as vs:
      self.input_len = input_len
      self.output_len = output_len
      self.batch_size = batch_size
      self.learning_rate = tf.Variable(float(learning_rate),trainable=False)
      self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate*decay_factor)
      self.global_step = tf.Variable(0,trainable=False)
      self.inputs = [tf.placeholder(tf.int32,shape=[None],name='input{}'.format(i)) for i in range(input_len)]
      self.decoder_inputs = [tf.placeholder(tf.int32,shape=[None],name='decoder_input{}'.format(i)) for i in range(output_len)]
      self.targets = self.decoder_inputs[1:]+[tf.placeholder(tf.int32,shape=[None],name='last_target')]
      self.target_weights = [tf.placeholder(tf.float32,shape=[None],name='target_weight{}'.format(i)) for i in range(output_len)]

      with tf.variable_scope('use_LM'):
        embeddings = tf.get_variable(name='embeddings',shape=[vocab_size,emb_dim],dtype=tf.float32,initializer=tf.random_uniform_initializer())

      emb_inps = [tf.nn.embedding_lookup(embeddings,ele) for ele in self.inputs]
      emb_decoder_inps = [tf.nn.embedding_lookup(embeddings,ele) for ele in self.decoder_inputs]

      cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(size) for i in range(num_layers)])
      #encoder_outputs,_ = tf.contrib.rnn.static_rnn(cell,emb_inps,dtype=tf.float32)
      
      top_states = [tf.reshape(ele,[-1,1,cell.output_size]) for ele in emb_inps]
      att_states = tf.concat(top_states,axis=1)

      common_states = tf.nn.conv2D(tf.expend_dims(att_states,-1),[5,5,1,2],[1,2,2,1],padding="SAME")
      common_states = tf.nn.relu(common_states)
      common_states = tf.nn.conv2D(common_states,[3,3,2,4],[1,2,2,1],padding="SAME")
      common_states = tf.nn.relu(common_states)
      common_states = tf.nn.pool(common_states,[3,3,4,8],[1,2,2,1],padding="SAME")
      common_states = tf.reshape(common_states,[self.batch_size,-1])
      
      att_length = att_states.get_shape()[1].value
      att_size = att_states.get_shape()[2].value
      att_vec_size = att_size

      hidden = tf.reshape(att_states,[-1,att_length,1,att_size])
      hidden_features = []
      v = []
      for a in range(num_heads):
        k = tf.get_variable('AttnW_%d'%a,[1,1,att_size,att_vec_size])
        hidden_features.append(tf.nn.conv2d(hidden,k,[1,1,1,1],'SAME'))
        v.append(tf.get_variable('AttnV_%d'%a,[att_vec_size]))

      def attention(query):
        """Put attention masks on hidden using hidden_features and query."""
        ds = []  # Results of attention reads will be stored here.
        #if tf.contrib.framework.nest.is_sequence(query):  # If the query is a tuple, flatten it.
        #  query_list = tf.contrib.framework.nest.flatten(query)
        #  for q in query_list:  # Check that ndims == 2 if specified.
        #    ndims = q.get_shape().ndims
        #    if ndims:
        #      assert ndims == 2
        #  query = tf.concat(query_list, 1)
        for a in xrange(num_heads):
          with tf.variable_scope("Attention_%d" % a):
            y = linear(query, att_vec_size, True)
            y = tf.reshape(y, [-1, 1, 1, att_vec_size])
            # Attention mask is a softmax of v^T * tanh(...).
            s = tf.reduce_sum(v[a] * tf.tanh(hidden_features[a] + y),
                                  [2, 3])
            a_ = tf.nn.softmax(s)
            # Now calculate the attention-weighted vector d.
            d = tf.reduce_sum(
                tf.reshape(a_, [-1, att_length, 1, 1]) * hidden, [1, 2])
            ds.append(tf.reshape(d, [-1, att_size]))
        return ds
      #decoder_cell = LM.cell
      with tf.variable_scope('loop'):
        decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(size) for i in range(num_layers)])
        state = decoder_cell.zero_state(LM.batch_size,dtype=tf.float32)
        #print(state)
        #state = tf.zeros([self.batch_size,size])
        attns = attention(state)
        #with tf.variable_scope('first_inp'):
        #  inp = linear(attns,emb_dim,True)
        outputs = []
        for i,inp in enumerate(emb_decoder_inps):
          if i>0:
            variable_scope.get_variable_scope().reuse_variables()
          x = linear([common_states]+[inp]+attns,emb_dim,True)
          with tf.variable_scope('use_LM'):
            cell_output,state = decoder_cell(x,state)
          with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                             reuse=True):
            attns = attention(state)
          with tf.variable_scope('AttnOutputProjection'):
            output = linear([cell_output]+attns,emb_dim,True)
          outputs.append(output)
      with tf.variable_scope('use_LM'):
        proj_w = tf.get_variable('proj_w',[emb_dim,vocab_size])
        proj_b = tf.get_variable('proj_b',[vocab_size])
        w_t = tf.transpose(proj_w)
      def nce_loss(labels,inputs):
        labels = tf.reshape(labels,[-1,1])
        return tf.nn.sampled_softmax_loss(w_t,proj_b,labels,inputs,num_samples,vocab_size)
      #self.losses = tf.reduce_mean(nce_loss(tf.concat(self.targets,axis=0),tf.concat(outputs,axis=0)))
      self.losses = sequence_loss(outputs,self.targets,self.target_weights,softmax_loss_function=nce_loss)
    self.predict = [tf.nn.xw_plus_b(ele,LM.proj_w,LM.proj_b) for ele in outputs]
    self.params = []
    self.params_use_LM = []
    for ele in tf.global_variables():
      if ele.name.startswith(name):
        if ele.name.find('use_LM')>-1:
          self.params_use_LM.append(ele)
        else:
          self.params.append(ele)
    #assign params of LM
    self.assign_params = [tf.assign(ele,LM.params[ind+2]) for ind, ele in enumerate(self.params_use_LM)]
    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
    gradients = tf.gradients(self.losses,self.params+self.params_use_LM)
    clipped_gradients,norm = tf.clip_by_global_norm(gradients,max_gradient_norm)
    self.update = opt.apply_gradients(zip(clipped_gradients,self.params+self.params_use_LM),global_step=self.global_step)
    self.saver = tf.train.Saver(self.params+self.params_use_LM)
    tmp = [ele.name for ele in self.params]
    print(len(tmp))
    print('\n'.join(tmp))
    tmp = [ele.name for ele in self.params_use_LM]
    print(len(tmp))
    print('\n'.join(tmp))

  def step(self,sess,encoder_inps,decoder_inps,weights,forward_only=False):
    if len(encoder_inps)!=self.input_len:
      raise ValueError('input length(%d) must be equal to the input_len(%d)'%(len(inputs),self.input_len))
    if len(weights)!=self.output_len:
      raise ValueError('weight length(%d) must be equal to the output_len(%d)'%(len(weights),self.output_len))
    if len(decoder_inps)!= self.output_len:
      raise ValueError('target length(%d) must be equal to the output_len(%d)'%(len(weights),self.output_len))
    input_feed = {}
    for i in range(self.input_len):
      input_feed[self.inputs[i].name] = encoder_inps[i]
    for i in range(self.output_len):
      input_feed[self.target_weights[i].name] = weights[i]
      input_feed[self.decoder_inputs[i].name] = decoder_inps[i]
    input_feed[self.targets[-1].name] = np.zeros(self.batch_size)
    #input_feed[self.targets[-1].name] = np.zeros([self.batch_size],dtype=np.int32)

    if not forward_only:
       output_feed = [self.losses,self.update]
    else:
       output_feed = [self.losses,self.predict]
    outputs = sess.run(output_feed,input_feed)
    return outputs[0],outputs[1]

  def get_batch(self, data):
    encoder_inputs, decoder_inputs = [], []
    for _ in range(self.batch_size):

        encoder_input, decoder_input = random.choice(data)
        # Encoder inputs are padded and then reversed.
        encoder_pad = [data_utils.PAD_ID] * (self.input_len - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = self.output_len - len(decoder_input)
        decoder_inputs.append(decoder_input +
                              [data_utils.PAD_ID] * decoder_pad_size)

    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in range(self.input_len):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))
    for length_idx in range(self.output_len):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in range(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx<self.output_len-1:
          target = decoder_inputs[batch_idx][length_idx+1]

        if length_idx==self.output_len-1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs,batch_weights
      
#if __name__ == '__main__':
  #vocab_size,emb_dim,size,num_steps,learning_rate,decay_factor,batch_size,num_samples=512
#   model = cnn2seq(100,100,100,100,1.0,0.99,100)
