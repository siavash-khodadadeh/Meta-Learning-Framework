import os

import tensorflow as tf
import numpy as np

import settings
from data.tf_datasets import OmniglotDataset, MiniImagenetDataset, AirCraftDataset
from meta_learning import ModelAgnosticMetaLearning
from models import SimpleModel, MAMLMiniImagenetModel


class Configuration(object):
    def __init__(
            self,
            model,
            meta_learning_dataset,
            meta_learning_val_dataset,
            meta_learning_iterations,
            meta_learning_save_after_iterations,
            meta_learning_log_after_iterations,
            meta_learning_summary_after_iterations,
            training_dataset,
            num_updates_meta_learning,
            num_updates_training,
            meta_batch_size,
            meta_learning_rate,
            learning_rate_meta_learning,
            learning_rate_training,
            n,
            k_meta_learning,
            k_training,
            meta_learning_log_dir,
            evaluation_log_dir,
            meta_learning_saving_path,
            training_saving_path=None,
            task_type='MAML_Supervised',
    ):
        self.meta_learning_iterations = meta_learning_iterations
        self.meta_learning_save_after_iterations = meta_learning_save_after_iterations
        self.meta_learning_log_after_iterations = meta_learning_log_after_iterations
        self.meta_learning_summary_after_iterations = meta_learning_summary_after_iterations
        self.model = model
        self.meta_learning_dataset = meta_learning_dataset
        self.meta_learning_val_dataset = meta_learning_val_dataset
        self.training_dataset = training_dataset
        self.num_updates_meta_learning = num_updates_meta_learning
        self.num_updates_training = num_updates_training
        self.meta_batch_size = meta_batch_size
        self.meta_learning_rate = meta_learning_rate
        self.learning_rate_meta_learning = learning_rate_meta_learning
        self.learning_rate_training = learning_rate_training
        self.n = n
        self.k_meta_learning = k_meta_learning
        self.k_training = k_training
        self.task_type = task_type

        if meta_learning_log_dir == 'config':
            self.meta_learning_log_dir = os.path.join(self.get_log_dir(), 'meta-learning')
        else:
            self.meta_learning_log_dir = meta_learning_log_dir
        if evaluation_log_dir == 'config':
            self.evaluation_log_dir = os.path.join(self.get_log_dir(), 'adaptation')
        else:
            self.evaluation_log_dir = evaluation_log_dir
        if meta_learning_saving_path == 'config':
            self.meta_learning_saving_path = os.path.join(self.get_log_dir(), 'saved_models')
        else:
            self.meta_learning_saving_path = meta_learning_saving_path

        # This part could be simplified. It is just because I wanted not to get IDE error
        # that I compare None with equality
        if training_saving_path is None:
            self.training_saving_path = None
        else:
            if training_saving_path == 'config':
                self.training_saving_path = os.path.join(self.get_log_dir(), 'saved_adaptation_models')
            else:
                self.training_saving_path = training_saving_path

    def get_log_dir(self):

        #  The model is not instantiated yet and does not have a name. We call the method with None as self in order to
        #  return the name of the model.
        log_dir = 'task-' + str(self.task_type) + '_' + \
                  'model-' + self.model.get_name(None) + '_' + \
                  'mldata-' + str(self.meta_learning_dataset) + '_' + \
                  'mliter-' + str(self.meta_learning_iterations) + '_' + \
                  'trdata-' + str(self.training_dataset) + '_' + \
                  'mlnupdates-' + str(self.num_updates_meta_learning) + '_' + \
                  'trnupdates-' + str(self.num_updates_training) + '_' + \
                  'mbs-' + str(self.meta_batch_size) + '_' + \
                  'mlr-' + str(self.meta_learning_rate) + '_' + \
                  'mllr-' + str(self.learning_rate_meta_learning) + '_' + \
                  'trlr-' + str(self.learning_rate_training) + '_' + \
                  'n-' + str(self.n) + '_' + \
                  'mlk-' + str(self.k_meta_learning) + '_' + \
                  'trk-' + str(self.k_training, ) + '_'
        return os.path.join(settings.DEFAULT_LOG_DIR, log_dir)

    def supervised_maml(self):

        val_tr_task, val_val_task, val_tr_labels, val_val_labels = \
            self.meta_learning_val_dataset.get_supervised_meta_learning_tasks(
                meta_batch_size=self.meta_batch_size,
                n=self.n,
                k=self.k_meta_learning,
            )

        train_task, val_task, train_labels, val_labels = self.meta_learning_dataset.get_supervised_meta_learning_tasks(
            meta_batch_size=self.meta_batch_size,
            n=self.n,
            k=self.k_meta_learning
        )
        maml = ModelAgnosticMetaLearning(
            self.model,
            train_task,
            val_task,
            train_labels,
            val_labels,
            val_dataset=(val_tr_task, val_val_task, val_tr_labels, val_val_labels),
            output_dimension=self.n,
            meta_learning_iterations=self.meta_learning_iterations,
            meta_learning_log_after_iterations=self.meta_learning_log_after_iterations,
            meta_learning_save_after_iterations=self.meta_learning_save_after_iterations,
            meta_learning_summary_after_iterations=self.meta_learning_summary_after_iterations,
            update_lr=self.learning_rate_meta_learning,
            meta_lr=self.meta_learning_rate,
            meta_batch_size=self.meta_batch_size,
            # stop_grad=False,
            num_updates=self.num_updates_meta_learning
        )

        file_writer = tf.summary.FileWriter(self.meta_learning_log_dir, tf.get_default_graph())
        validation_file_writer = tf.summary.FileWriter(os.path.join(self.meta_learning_log_dir, 'validation'))
        maml.meta_learn(file_writer, validation_file_writer, saving_path=self.meta_learning_saving_path)

    def evaluate_supervised_maml(self):
        model_address = os.path.join(self.meta_learning_saving_path)
        training_model = self.model(output_dimension=self.n, update_lr=self.learning_rate_training)
        train_task, val_task, train_labels, val_labels = self.training_dataset.get_supervised_meta_learning_tasks(
            meta_batch_size=1,
            n=self.n,
            k=self.k_training
        )

        tf.summary.image('task', tf.reshape(train_task, training_model.get_input_shape()), max_outputs=12)

        training_model.forward(train_task)
        training_model.define_update_op(train_labels, with_batch_norm_dependency=True)
        training_model.define_accuracy(train_labels)

        for item in tf.global_variables():
            tf.summary.histogram(item.name, item)

        merged_summary = tf.summary.merge_all()

        with tf.Session() as sess:
            self.summary_evaluation_task_number = 100
            self.num_tasks_evaluation = 500
            task_losses = np.zeros(shape=(self.num_tasks_evaluation, self.num_updates_training))
            task_outs = np.zeros(shape=(self.num_tasks_evaluation, self.num_updates_training, 5))
            task_accuracies = np.zeros(shape=(self.num_tasks_evaluation, self.num_updates_training))

            for task in range(self.num_tasks_evaluation):
                train_task_np, val_task_np, train_labels_np, val_labels_np = sess.run(
                    (train_task, val_task, train_labels, val_labels)
                )

                if task % self.summary_evaluation_task_number == 0:
                    train_writer = tf.summary.FileWriter(
                        os.path.join(self.evaluation_log_dir, 'training', 'task-num_{}'.format(task)),
                    )
                    test_writer = tf.summary.FileWriter(
                        os.path.join(self.evaluation_log_dir, 'test', 'task-num{}'.format(task))
                    )

                sess.run(tf.global_variables_initializer())
                training_model.load(sess, model_address)
                for it in range(self.num_updates_training):
                    output, loss, acc, summ = sess.run(
                        (training_model.out, training_model.loss, training_model.accuracy, merged_summary),
                        feed_dict={
                            training_model.is_training: False,
                            train_task: val_task_np,
                            train_labels: val_labels_np,
                        }
                    )
                    if task % self.summary_evaluation_task_number == 0:
                        test_writer.add_summary(summ, global_step=it)
                    _, summ = sess.run((training_model.op, merged_summary), feed_dict={
                        training_model.is_training: True,
                        train_task: train_task_np,
                        train_labels: train_labels_np
                    })
                    if task % self.summary_evaluation_task_number == 0:
                        train_writer.add_summary(summ, global_step=it)

                    if task % 100 == 0:
                        task_losses[task, it] = loss
                        print(loss)
                        task_accuracies[task, it] = acc
                        print(acc)
                        task_outs[task, it, :] = np.argmax(output, 1)
                        print(np.argmax(output, 1))

            print(task_losses)
            print(task_outs)
            print(task_accuracies)
            print('done')

            print('average loss:')
            print(np.mean(task_losses, axis=0))

            print('average accuracy:')
            print(np.mean(task_accuracies, axis=0))


    def meta_learning(self):
        if self.task_type == 'MAML_Supervised':
            self.supervised_maml()

    def evaluate(self):
        tf.reset_default_graph()
        if self.task_type == 'MAML_Supervised':
            self.evaluate_supervised_maml()

    def execute(self):
        self.meta_learning()
        self.evaluate()


if __name__ == '__main__':
    omniglot_dataset = OmniglotDataset()
    config = Configuration(
        model=SimpleModel,
        meta_learning_dataset=omniglot_dataset.get_train_dataset(),
        meta_learning_val_dataset=omniglot_dataset.get_validation_dataset(),
        meta_learning_iterations=15001,
        meta_learning_save_after_iterations=5000,
        meta_learning_log_after_iterations=100,
        meta_learning_summary_after_iterations=100,
        training_dataset=omniglot_dataset.get_test_dataset(),
        num_updates_meta_learning=5,
        num_updates_training=15,
        meta_batch_size=1,
        meta_learning_rate=0.001,
        learning_rate_meta_learning=0.1,
        learning_rate_training=0.1,
        n=5,
        k_meta_learning=1,
        k_training=1,
        meta_learning_log_dir='config',
        evaluation_log_dir='config',
        meta_learning_saving_path='config',
        training_saving_path=None,
        task_type='MAML_Supervised',
    )
    config.meta_learning()
    config.evaluate()

    # miniimagenet_dataset = MiniImagenetDataset()
    # config = Configuration(
    #     model=MAMLMiniImagenetModel,
    #     meta_learning_dataset=miniimagenet_dataset.get_train_dataset(),
    #     meta_learning_val_dataset=miniimagenet_dataset.get_validation_dataset(),
    #     meta_learning_iterations=2001,
    #     meta_learning_save_after_iterations=1000,
    #     meta_learning_log_after_iterations=5,
    #     meta_learning_summary_after_iterations=1,
    #     training_dataset=miniimagenet_dataset.get_test_dataset(),
    #     num_updates_meta_learning=1,
    #     num_updates_training=15,
    #     meta_batch_size=3,
    #     meta_learning_rate=0.001,
    #     learning_rate_meta_learning=0.1,
    #     learning_rate_training=0.1,
    #     n=5,
    #     k_meta_learning=1,
    #     k_training=1,
    #     meta_learning_log_dir='config',
    #     evaluation_log_dir='config',
    #     meta_learning_saving_path='config',
    #     training_saving_path=None,
    #     task_type='MAML_Supervised',
    # )
    # config.meta_learning()
    # config.evaluate()

    # aircraft_dataset = AirCraftDataset()
    # config = Configuration(
    #     model=MAMLMiniImagenetModel,
    #     meta_learning_dataset=aircraft_dataset.get_train_dataset(),
    #     meta_learning_val_dataset=aircraft_dataset.get_validation_dataset(),
    #     meta_learning_iterations=2001,
    #     meta_learning_save_after_iterations=1000,
    #     meta_learning_log_after_iterations=5,
    #     meta_learning_summary_after_iterations=1,
    #     training_dataset=aircraft_dataset.get_test_dataset(),
    #     num_updates_meta_learning=1,
    #     num_updates_training=15,
    #     meta_batch_size=3,
    #     meta_learning_rate=0.001,
    #     learning_rate_meta_learning=0.1,
    #     learning_rate_training=0.1,
    #     n=5,
    #     k_meta_learning=1,
    #     k_training=1,
    #     meta_learning_log_dir='config',
    #     evaluation_log_dir='config',
    #     meta_learning_saving_path='config',
    #     training_saving_path=None,
    #     task_type='MAML_Supervised',
    # )
    # config.meta_learning()
    # config.evaluate()
