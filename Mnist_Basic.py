
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


from tensorflow.examples.tutorials.mnist import input_data


# In[3]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[4]:


type(mnist)


# In[5]:


mnist.train.images


# In[6]:


mnist.train.num_examples


# In[7]:


mnist.test.num_examples


# In[9]:


mnist.validation.num_examples


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


single_image = mnist.train.images[1].reshape(28,28)


# In[18]:


plt.imshow(single_image, cmap = 'gist_gray')


# In[19]:


single_image.min()


# In[20]:


single_image.max()


# In[23]:


# Placeholders
x = tf.placeholder(tf.float32, shape = [None, 784])


# In[25]:


# Variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# In[26]:


# Create Graph Operations
y = tf.matmul(x,W) + b


# In[27]:


# Loss Function
y_true = tf.placeholder(tf.float32, [None, 10])


# In[28]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y))


# In[30]:


#Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)
train = optimizer.minimize(cross_entropy)


# In[31]:


# Create Session
init = tf.global_variables_initializer()


# In[34]:


with tf.Session() as sess:
    sess.run(init)
    
    for step in range(1000):
        #train in batches
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict = {x:batch_x, y_true:batch_y})
        
    # Evaluate The Model
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_true, 1))
    
    #[true, false, true,...] ---> [1,0,1...]    
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Predicted [3,4]  True [3, 9]
    # [True, False]
    # [1.0, 0.0]
    # Average of that which is 0.5 or 50% accuracy
    
    print(sess.run(acc, feed_dict={x:mnist.test.images, y_true:mnist.test.labels}))

