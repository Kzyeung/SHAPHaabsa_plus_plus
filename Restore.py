"""
import tensorflow as tf

# Create a new session
sess = tf.Session()

# Import the graph definition from the .meta file
saver = tf.train.import_meta_graph('C:/Users/Kelvin/Desktop/Thesis/Prior Papers/HAABSA_plusplus/venv/trainedModelMaria/2016/-18800.meta')

# Restore the saved variables from the checkpoint file
saver.restore(sess, 'C:/Users/Kelvin/Desktop/Thesis/Prior Papers/HAABSA_plusplus/venv/trainedModelMaria/2016/checkpoint')

# Get a handle to the input tensor
input_tensor = sess.graph.get_tensor_by_name('input:0')

# Get a handle to the output tensor
output_tensor = sess.graph.get_tensor_by_name('output:0')

# Do some prediction with the model
input_data = [[1, 2, 3], [4, 5, 6]]
output_data = sess.run(output_tensor, feed_dict={input_tensor: input_data})
print(output_data)
"""

"""
import tensorflow as tf

# Load the saved model
model_path = 'C:/Users/Kelvin/Desktop/Thesis/Prior Papers/HAABSA_plusplus/venv/trainedModelMaria/2016/-18800'
with tf.Session() as sess:
    # Restore the graph from the saved model
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)
    # Get the graph_def object
    graph_def = sess.graph_def
    print("HERE WE GO")
    # Get the names of the tensors in the graph
    tensor_names = [tensor.name for tensor in graph_def.node]
    print(tensor_names)
"""

import tensorflow as tf

# Load the model from the .meta file
tf.reset_default_graph()
saver = tf.train.import_meta_graph('C:/Users/Kelvin/Desktop/Thesis/Prior Papers/HAABSA_plusplus/venv/trainedModelMaria/2016/-18800.meta')

# Start a session and restore the saved variables
with tf.Session() as sess:
    saver.restore(sess, 'C:/Users/Kelvin/Desktop/Thesis/Prior Papers/HAABSA_plusplus/venv/trainedModelMaria/2016/-18800')

    # Get the tensors from the loaded graph
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('inputs/Placeholder:0')
    y_pred = graph.get_tensor_by_name('inputs/Placeholder_1:0')
    is_training = graph.get_tensor_by_name('Placeholder:0')

    # Make predictions on new data
    x_new = 'C:/Users/Kelvin/Desktop/Thesis/Prior Papers/HAABSA_plusplus/venv/data/programGeneratedData/768remainingtestdata2016.txt' # Prepare new data as a numpy array
    #predictions = sess.run(y_pred, feed_dict={x: x_new})
    predictions = sess.run(y_pred, feed_dict={x: x_new, is_training: False})