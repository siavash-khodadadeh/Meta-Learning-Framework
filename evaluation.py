import tensorflow as tf
import numpy as np

from data.tf_datasets import OmniglotDataset
from models import SimpleModel


model_address = './saved_models/simple_model-1000'
model = SimpleModel()

omniglot_dataset = OmniglotDataset()
test_dataset = omniglot_dataset.get_test_dataset()
train_task, val_task, train_labels, val_labels = test_dataset.get_supervised_meta_learning_tasks(
    meta_batch_size=1,
    n=6,
    k=2
)

tf.summary.image('task', tf.reshape(train_task, (-1, 28, 28, 1)), max_outputs=12)

model.forward(train_task)
model.define_update_op(train_labels, with_batch_norm_dependency=True)


for item in tf.global_variables():
    tf.summary.histogram(item.name, item)

merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./adaptaion_summary/train', tf.get_default_graph())
test_writer = tf.summary.FileWriter('./adaptaion_summary/test')


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_task_np, val_task_np, train_labels_np, val_labels_np = sess.run(
        (train_task, val_task, train_labels, val_labels)
    )

    model.load(sess, model_address)

    for it in range(15):
        output, loss, summ = sess.run((model.out, model.loss, merged_summary), feed_dict={
            model.is_training: False,
            train_task: val_task_np,
            train_labels: val_labels_np,
        })
        test_writer.add_summary(summ, global_step=it)
        _, summ = sess.run((model.op, merged_summary), feed_dict={
            model.is_training: True,
            train_task: train_task_np,
            train_labels: train_labels_np
        })
        train_writer.add_summary(summ, global_step=it)

        print(loss)
        print(np.argmax(output, 1))

    print('done')
