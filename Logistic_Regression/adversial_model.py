import tensorflow.compat.v1 as tf
from load_dataset import *
from fairness_metrics import *


class AdversarialLogisticModel(object):
    """A model for doing adversarial training of logistic models."""

    def __init__(self,
                 scope_name,
                 sess,
                 hyperparameters,
                 classifier_num_hidden_units=200,
                 seed=None,
                 debias=False):

        ii32 = np.iinfo(np.int32)
        self.seed1, self.seed2, self.seed3, self.seed4 = np.random.randint(ii32.min, ii32.max, size=4)
        self.sess = sess
        self.scope_name = scope_name
        self.seed = seed

        self.adversary_loss_weight = hyperparameters['adversary_loss_weight']
        self.num_epochs = hyperparameters["num_epochs"]
        self.batch_size = hyperparameters["batch_size"]
        self.debias = debias
        self.classifier_num_hidden_units = classifier_num_hidden_units

        self.features_dim = None
        self.features_ph = None
        self.protected_attributes_ph = None
        self.true_labels_ph = None
        self.pred_labels = None

    def predictor_model(self, features, features_dim):
        with tf.variable_scope("predictor_model"):
            W1 = tf.get_variable('W1', [self.features_dim, 1], initializer=tf.glorot_uniform_initializer())
            b1 = tf.Variable(tf.zeros(shape=[1]), name='b1')

            pred_logits = tf.matmul(features, W1) + b1
            pred_labels = tf.sigmoid(pred_logits)

        return pred_labels, pred_logits

    def _classifier_model(self, features, features_dim, keep_prob):
        """Compute the classifier predictions for the outcome variable.
        """
        with tf.variable_scope("classifier_model"):
            W1 = tf.get_variable('W1', [features_dim, self.classifier_num_hidden_units],
                                 initializer=tf.initializers.glorot_uniform(seed=self.seed1))
            b1 = tf.Variable(tf.zeros(shape=[self.classifier_num_hidden_units]), name='b1')

            h1 = tf.nn.relu(tf.matmul(features, W1) + b1)
            h1 = tf.nn.dropout(h1, keep_prob=keep_prob, seed=self.seed2)

            W2 = tf.get_variable('W2', [self.classifier_num_hidden_units, 1],
                                 initializer=tf.initializers.glorot_uniform(seed=self.seed3))
            b2 = tf.Variable(tf.zeros(shape=[1]), name='b2')

            pred_logit = tf.matmul(h1, W2) + b2
            pred_label = tf.sigmoid(pred_logit)

        return pred_label, pred_logit

    def _adversary_model(self, pred_logits, true_labels):
        """Compute the adversary predictions for the protected attribute.
        """
        with tf.variable_scope("adversary_model"):
            c = tf.get_variable('c', initializer=tf.constant(1.0))
            s = tf.sigmoid((1 + tf.abs(c)) * pred_logits)

            W2 = tf.get_variable('W2', [3, 1],
                                 initializer=tf.initializers.glorot_uniform(seed=self.seed4))
            b2 = tf.Variable(tf.zeros(shape=[1]), name='b2')

            pred_protected_attribute_logit = tf.matmul(tf.concat([s, s * true_labels, s * (1.0 - true_labels)], axis=1),
                                                       W2) + b2
            pred_protected_attribute_label = tf.sigmoid(pred_protected_attribute_logit)

        return pred_protected_attribute_label, pred_protected_attribute_logit

    def fit(self, features, labels, protect):
        if self.seed is not None:
            np.random.seed(self.seed)

        with tf.variable_scope(self.scope_name):
            num_train_samples, self.features_dim = np.shape(features)

            # Setup placeholders
            self.features_ph = tf.placeholder(tf.float32, shape=[None, self.features_dim])
            self.protected_attributes_ph = tf.placeholder(tf.float32, shape=[None, 1])
            self.true_labels_ph = tf.placeholder(tf.float32, shape=[None, 1])
            self.keep_prob = tf.placeholder(tf.float32)

            # Obtain classifier predictions and classifier loss
            self.pred_labels, pred_logits = self._classifier_model(self.features_ph, self.features_dim, self.keep_prob)
            pred_labels_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_labels_ph, logits=pred_logits))

            if self.debias:
                # Obtain adversary predictions and adversary loss
                pred_protected_attributes_labels, pred_protected_attributes_logits = self._adversary_model(
                    pred_logits, self.true_labels_ph)
                pred_protected_attributes_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.protected_attributes_ph,
                                                            logits=pred_protected_attributes_logits))

            # Setup optimizers with learning rates
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.001
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
            classifier_opt = tf.train.AdamOptimizer(learning_rate)
            if self.debias:
                adversary_opt = tf.train.AdamOptimizer(learning_rate)

            classifier_vars = [var for var in tf.trainable_variables() if 'classifier_model' in var.name]
            if self.debias:
                adversary_vars = [var for var in tf.trainable_variables() if 'adversary_model' in var.name]
                # Compute gradient of adversarial loss with respect to predictor variables
                adversary_grads = {var: grad for (grad, var) in
                                   adversary_opt.compute_gradients(pred_protected_attributes_loss,
                                                                   var_list=classifier_vars)}

            normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)
            classifier_grads = []
            for (grad, var) in classifier_opt.compute_gradients(pred_labels_loss, var_list=classifier_vars):
                if self.debias:
                    unit_adversary_grad = normalize(adversary_grads[var])
                    grad -= tf.reduce_sum(grad * unit_adversary_grad) * unit_adversary_grad
                    grad -= self.adversary_loss_weight * adversary_grads[var]
                classifier_grads.append((grad, var))
            # Update predictor parameters
            classifier_minimizer = classifier_opt.apply_gradients(classifier_grads, global_step=global_step)

            if self.debias:
                # Update adversary parameters
                with tf.control_dependencies([classifier_minimizer]):
                    adversary_minimizer = adversary_opt.minimize(pred_protected_attributes_loss,
                                                                 var_list=adversary_vars,
                                                                 global_step=global_step)

            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

            # Begin training
            for epoch in range(self.num_epochs):
                shuffled_ids = np.random.choice(num_train_samples, num_train_samples)
                for i in range(num_train_samples // self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)]
                    batch_features = features[batch_ids]
                    batch_labels = np.reshape(labels[batch_ids], [-1, 1])
                    batch_protected_attributes = np.reshape(protect[batch_ids], [-1, 1])

                    batch_feed_dict = {self.features_ph: batch_features,
                                       self.true_labels_ph: batch_labels,
                                       self.protected_attributes_ph: batch_protected_attributes,
                                       self.keep_prob: 0.8}
                    if self.debias:
                        _, _, pred_labels_loss_value, pred_protected_attributes_loss_vale = self.sess.run(
                            [classifier_minimizer, adversary_minimizer,
                             pred_labels_loss, pred_protected_attributes_loss], feed_dict=batch_feed_dict)
                        if i % 200 == 0:
                            print("epoch %d; iter: %d; batch predictor loss: %f; batch adversarial loss: %f" % (
                                epoch, i, pred_labels_loss_value,
                                pred_protected_attributes_loss_vale))
                    else:
                        _, pred_labels_loss_value = self.sess.run([classifier_minimizer, pred_labels_loss],
                                                                  feed_dict=batch_feed_dict)
                        if i % 200 == 0:
                            print("epoch %d; iter: %d; batch classifier loss: %f" % (
                                epoch, i, pred_labels_loss_value))
        return self

    def predict(self, features, labels, protect):
        """Obtain the predictions for the provided dataset using the fair classifier learned.
        Args:
            features
            labels
            protect
        Returns:
            predictions
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        num_test_samples, _ = np.shape(features)

        samples_covered = 0
        pred_labels = []
        while samples_covered < num_test_samples:
            start = samples_covered
            end = samples_covered + self.batch_size
            if end > num_test_samples:
                end = num_test_samples
            batch_ids = np.arange(start, end)
            batch_features = features[batch_ids]
            batch_labels = np.reshape(labels[batch_ids], [-1, 1])
            batch_protected_attributes = np.reshape(protect[batch_ids], [-1, 1])

            batch_feed_dict = {self.features_ph: batch_features,
                               self.true_labels_ph: batch_labels,
                               self.protected_attributes_ph: batch_protected_attributes,
                               self.keep_prob: 1.0}

            pred_labels += self.sess.run(self.pred_labels, feed_dict=batch_feed_dict)[:, 0].tolist()
            samples_covered += len(batch_features)

        true_labels = list(labels.reshape(1, -1)[0])

        pred_labels = np.array([1 if i > 0.5 else 0 for i in pred_labels]).reshape(-1, 1)

        return pred_labels


filepath = "../Datasets/adult_dataset/processed_adult.csv"
label = "income"
protect = ["gender"]
num_proxy_to_remove = 0
train_dataset, test_dataset, _ = train_test_dataset(filepath,
                                                    label,
                                                    protect,
                                                    is_scaled=True,
                                                    num_proxy_to_remove=num_proxy_to_remove,
                                                    balanced=None
                                                    )

train_features, train_labels, train_protect = train_dataset.features, train_dataset.label, train_dataset.protect
test_features, test_labels, test_protect = test_dataset.features, test_dataset.label, test_dataset.protect

hyperparameters = {'adversary_loss_weight': 0.1,
                   'batch_size': 128,
                   'num_epochs': 50,
                   }
with tf.Session() as sess:
    model = AdversarialLogisticModel("training", sess, hyperparameters, debias=True)
    trained_model = model.fit(train_features, train_labels, train_protect)
    train_pred_labels = trained_model.predict(train_features, train_labels, train_protect)
    test_pred_labels = trained_model.predict(test_features, test_labels, test_protect)

accs = equalizing_odds(test_pred_labels, test_labels, test_protect)
print("Train Accuracy: ", accuracy_score(train_labels, train_pred_labels))
print("Test Accuracy: ", accuracy_score(test_labels, test_pred_labels))
print(accs)
print([max(acc) - min(acc) for acc in accs])
