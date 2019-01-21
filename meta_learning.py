import tensorflow as tf

from utils import xent


class ModelAgnosticMetaLearning(object):
    def __init__(
        self,
        model,
        train_task,
        val_task,
        train_labels,
        val_labels,
        val_dataset,
        output_dimension,
        meta_learning_iterations,
        meta_learning_log_after_iterations,
        meta_learning_save_after_iterations,
        meta_learning_summary_after_iterations,
        update_lr,
        meta_lr,
        meta_batch_size,
        num_updates
    ):
        self.meta_learning_iterations = meta_learning_iterations
        self.meta_learning_log_after_iterations = meta_learning_log_after_iterations
        self.meta_learning_save_after_iterations = meta_learning_save_after_iterations
        self.meta_learning_summary_after_iterations = meta_learning_summary_after_iterations
        self.val_tr_task, self.val_val_task, self.val_tr_labels, self.val_val_labels = val_dataset
        self.model = model(output_dimension=output_dimension, update_lr=update_lr)
        self.train_task = train_task
        self.validation_task = val_task
        self.train_labels = train_labels
        self.validation_labels = val_labels
        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.meta_batch_size = meta_batch_size
        self.stop_grad = False
        self.num_updates = num_updates

        self.optimizer = tf.train.AdamOptimizer(self.meta_lr)
        self.meta_train_op = self.construct_meta_learn()

        for item in tf.global_variables():
            tf.summary.histogram(item.name, item)
        self.summary_merge_op = tf.summary.merge_all()

    def inner_loss_function(self, predictions, labels):
        return tf.reduce_mean(xent(predictions, labels))

    def outer_loss_function(self, predictions, labels):
        return tf.reduce_mean(xent(predictions, labels))

    def construct_meta_learn(self):
        tr_inp = tf.split(self.train_task, self.meta_batch_size)
        tf.summary.image('train', tf.reshape(self.train_task[0, :, :], self.model.get_input_shape()), max_outputs=12)
        val_inp = tf.split(self.validation_task, self.meta_batch_size)
        tf.summary.image('val', tf.reshape(self.validation_task[0, :, :], self.model.get_input_shape()), max_outputs=12)
        tr_labels = tf.split(self.train_labels, self.meta_batch_size)
        val_labels = tf.split(self.validation_labels, self.meta_batch_size)

        meta_batch_losses = []

        for inputa, labelsa, inputb, labelsb in zip(tr_inp, tr_labels, val_inp, val_labels):
            fast_weights = {key: val for key, val in self.model.weights.items()}
            outputs = []
            losses = []
            for j in range(self.num_updates):
                with tf.variable_scope('update', reuse=tf.AUTO_REUSE):
                    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
                        outputa = self.model.forward(inputa, fast_weights)
                    with tf.variable_scope('inner_loss'):
                        lossa = self.inner_loss_function(outputa, labelsa)

                    with tf.variable_scope('inner_gradients'):
                        grads = tf.gradients(lossa, list(fast_weights.values()))
                        if self.stop_grad:
                            grads = [tf.stop_gradient(grad) for grad in grads]

                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(
                        fast_weights.keys(),
                        [fast_weights[key] - self.update_lr * gradients[key] for key in fast_weights.keys()]
                    ))
                    with tf.variable_scope('model', reuse=True):
                        outputb = self.model.forward(inputb, fast_weights)
                    with tf.variable_scope('meta_loss'):
                        lossb = self.outer_loss_function(outputb, labelsb)

                    outputs.append(outputb)
                    losses.append(lossb)

            meta_batch_losses.append(losses[-1])

        self.meta_loss = tf.add_n(meta_batch_losses)
        tf.summary.scalar('meta_loss', self.meta_loss)

        meta_train_op = self.optimizer.minimize(self.meta_loss)
        return meta_train_op

    def meta_learn(self, file_writer, validation_file_writer, saving_path):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for it in range(self.meta_learning_iterations):
                if it % self.meta_learning_log_after_iterations == 0:
                    val_tr_task_np, val_val_task_np, val_tr_labels_np, val_val_labels_np= sess.run(
                        (self.val_tr_task, self.val_val_task, self.val_tr_labels, self.val_val_labels)
                    )

                    val_meta_loss, val_summ = sess.run((self.meta_loss, self.summary_merge_op), feed_dict={
                        self.model.is_training: True,
                        self.train_task: val_tr_task_np,
                        self.train_labels: val_tr_labels_np,
                        self.validation_task: val_val_task_np,
                        self.validation_labels: val_val_labels_np
                    })
                    validation_file_writer.add_summary(val_summ, it)
                    print('Loss for iteration {it}: {loss}'.format(it=it, loss=val_meta_loss))

                if it % self.meta_learning_summary_after_iterations == 0:
                    merged_summary = sess.run(self.summary_merge_op, feed_dict={self.model.is_training: True})
                    file_writer.add_summary(merged_summary, it)

                sess.run(self.meta_train_op, feed_dict={self.model.is_training: True})

                if it != 0 and it % self.meta_learning_save_after_iterations == 0:
                    self.model.save(sess, path=saving_path, step=it)
