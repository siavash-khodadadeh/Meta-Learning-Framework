import tensorflow as tf
import unittest

from data.tf_datasets import OmniglotDataset


class TestDataset(unittest.TestCase):
    def setUp(self):
        def _parse_function(example_address):
            return example_address

        omniglot_dataset = OmniglotDataset(parse_function=_parse_function)
        val_dataset = omniglot_dataset.get_train_dataset()

        train_task, val_task, train_labels, val_labels = val_dataset.get_supervised_meta_learning_tasks(
            meta_batch_size=3,
            n=6,
            k=2
        )

        self.train_task = train_task
        self.val_task = val_task
        self.train_labels = train_labels
        self.val_labels = val_labels

    def test_train_and_val_tasks_are_the_same(self):
        print('Test train and val task are the same!')
        with tf.Session() as sess:
            for experiment in range(10000):
                if experiment % 100 == 0:
                    print(experiment)

                tr_np, val_np, tr_lb_np, val_lb_np = sess.run(
                    (self.train_task, self.val_task, self.train_labels, self.val_labels)
                )

                for x in range(3):
                    for index in range(2 * 6):
                        tr_element = str(tr_np[x, index, 0])
                        val_element = str(val_np[x, index, 0])
                        self.assertEqual(tr_element[:tr_element.rfind('/')], val_element[:val_element.rfind('/')])

    def test_no_two_same_classes_in_the_same_task(self):
        print('Test no two same classes are sampled in the same task!')
        with tf.Session() as sess:
            for experiment in range(10000):
                if experiment % 1 == 0:
                    print(experiment)

                tr_np, val_np, tr_lb_np, val_lb_np = sess.run(
                    (self.train_task, self.val_task, self.train_labels, self.val_labels)
                )

                print(tr_np)
                for x in range(3):
                    classes_count = dict()
                    for index in range(2 * 6):
                        tr_element = str(tr_np[x, index, 0])
                        tr_element = tr_element[:tr_element.rindex('/')]
                        tr_element = tr_element[tr_element.rindex('/'):]
                        if tr_element in classes_count.keys():
                            classes_count[tr_element] += 1
                        else:
                            classes_count[tr_element] = 1

                    for class_name, count in classes_count.items():
                        self.assertEquals(count, 2)

    def test_classes_sampled_with_equal_chance_in_each_epoch(self):
        print('Test if class sampling is normal')
        classes_count = dict()

        with tf.Session() as sess:
            for experiment in range(30):

                tr_np, val_np, tr_lb_np, val_lb_np = sess.run(
                    (self.train_task, self.val_task, self.train_labels, self.val_labels)
                )

                for x in range(3):
                    for index in range(2 * 6):
                        tr_element = str(tr_np[x, index, 0])
                        tr_element = tr_element[:tr_element.rindex('/')]
                        tr_element = tr_element[tr_element.rindex('/'):]

                        if tr_element in classes_count.keys():
                            classes_count[tr_element] += 1
                        else:
                            classes_count[tr_element] = 1
        print(classes_count)
        print(len(classes_count))

    def test_different_samples_from_each_class_are_chosen(self):
        print('Test if instance sampling is normal')
        samples_count = dict()

        with tf.Session() as sess:
            for experiment in range(500):
                if experiment % 100 == 0:
                    print(experiment)

                tr_np, val_np, tr_lb_np, val_lb_np = sess.run(
                    (self.train_task, self.val_task, self.train_labels, self.val_labels)
                )

                for x in range(3):
                    for index in range(2 * 6):
                        tr_element = str(tr_np[x, index, 0])
                        if tr_element in samples_count.keys():
                            samples_count[tr_element] += 1
                        else:
                            samples_count[tr_element] = 1
        print(samples_count)
        print(len(samples_count))
        print()


if __name__ == '__main__':
    unittest.main()
