import tensorflow as tf

#Building a Graph
#As we said before, TensorFlow works as a graph computational model. Let's create our first graph which we named as graph1.

graph1 = tf.Graph()
#Now we call the TensorFlow functions that construct new tf.Operation and tf.Tensor objects and add them to the graph1. As mentioned, each tf.Operation is a node and each tf.Tensor is an edge in the graph.

#Lets add 2 constants to our graph. For example, calling tf.constant([2], name = 'constant_a') adds a single tf.Operation to the default graph. This operation produces the value 2, and returns a tf.Tensor that represents the value of the constant.
#Notice: tf.constant([2], name="constant_a") creates a new tf.Operation named "constant_a" and returns a tf.Tensor named "constant_a:0".

with graph1.as_default():
    a = tf.constant([2], name = 'constant_a')
    b = tf.constant([3], name = 'constant_b')
#Lets look at the tensor a.

print(a)

#After that, let's make an operation over these tensors. The function tf.add() adds two tensors (you could also use c = a + b).

with graph1.as_default():
    c = tf.add(a, b)
    #c = a + b is also a way to define the sum of the terms
    
#Then TensorFlow needs to initialize a session to run our code. Sessions are, in a way, a context for creating a graph inside TensorFlow. Let's define our session:

sess = tf.Session(graph = graph1)
#Let's run the session to get the result from the previous defined 'c' operation:

result = sess.run(c)
print(result)

sess.close()


#To avoid having to close sessions every time, we can define them in a with block, so after running the with block the session will close automatically:

with tf.Session(graph = graph1) as sess:
    result = sess.run(c)
    print(result)
    
#Defining multidimensional arrays using TensorFlow
#Now we will try to define such arrays using TensorFlow:
graph2 = tf.Graph()
with graph2.as_default():
    Scalar = tf.constant(2)
    Vector = tf.constant([5,6,2])
    Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
    Tensor = tf.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )
with tf.Session(graph = graph2) as sess:
    result = sess.run(Scalar)
    print ("Scalar (1 entry):\n %s \n" % result)
    result = sess.run(Vector)
    print ("Vector (3 entries) :\n %s \n" % result)
    result = sess.run(Matrix)
    print ("Matrix (3x3 entries):\n %s \n" % result)
    result = sess.run(Tensor)
    print ("Tensor (3x3x3 entries) :\n %s \n" % result)
#tf.shape returns the shape of our data structure.

print(Scalar.shape)
print(Tensor.shape)

#We then need to use another TensorFlow function called tf.matmul():

graph4 = tf.Graph()
with graph4.as_default():
    Matrix_one = tf.constant([[2,3],[3,4]])
    Matrix_two = tf.constant([[2,3],[3,4]])

    mul_operation = tf.matmul(Matrix_one, Matrix_two)

with tf.Session(graph = graph4) as sess:
    result = sess.run(mul_operation)
    print ("Defined using tensorflow function :")
    print(result)

# Now that you understand these data structures, I encourage you to play with them using some previous functions to see how they will behave, according to their structure types:    
graph3 = tf.Graph()
with graph3.as_default():
    Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
    Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

    add_1_operation = tf.add(Matrix_one, Matrix_two)
    add_2_operation = Matrix_one + Matrix_two

with tf.Session(graph =graph3) as sess:
    result = sess.run(add_1_operation)
    print ("Defined using tensorflow function :")
    print(result)
    result = sess.run(add_2_operation)
    print ("Defined using normal expressions :")
    print(result)   

    

