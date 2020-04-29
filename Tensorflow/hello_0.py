import tensorflow as tf


def test_graph():
    g1 = tf.Graph()
    with g1.as_default():
        # 定义变量v并初始化
        v = tf.get_variable("v", shape=[1], initializer=tf.zeros_initializer)

    g2 = tf.Graph()
    with g2.as_default():
        v = tf.get_variable("v", shape=[1], initializer=tf.ones_initializer)

    with tf.Session(graph=g2) as sess:
        tf.global_variables_initializer().run()
        with tf.variable_scope("", reuse=True):
            print(sess.run(tf.get_variable("v")))

    with g1.device('/cpu:0'):
        a = 0
        b = 0
        result = a+b

    tf.add_to_collection
    tf.get_collection


def test_tensor():
    a = tf.constant([1.0, 2.0], name='a')
    b = tf.constant([2.0, 3.0], name='b')
    result = tf.add(a, b, name='add')
    print(result)

# test_tensor()


def test_session():
    sess = tf.Session()
    sess.run()
    sess.close()
    with tf.Session() as sess:
        sess.run()

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess = tf.Session(config=config)
    sess.graph.get_tensor_by_name()
