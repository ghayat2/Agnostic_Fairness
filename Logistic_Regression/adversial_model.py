import tensorflow.compat.v1 as tf

from load_dataset import *
from fairness_metrics import *
import getopt
import sys
import os


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

    def _classifier_model(self, features, features_dim, keep_prob):
        """Compute the classifier predictions for the outcome variable.
        """
        with tf.variable_scope("classifier_model", reuse=tf.AUTO_REUSE):
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

    def _classifier_model_LR(self, features, features_dim, keep_prob):
        with tf.variable_scope("classifier_model"):
            W1 = tf.get_variable('W1', [features_dim, 1], initializer=tf.glorot_uniform_initializer())
            b1 = tf.Variable(tf.zeros(shape=[1]), name='b1')

            pred_logits = tf.matmul(features, W1) + b1
            pred_labels = tf.sigmoid(pred_logits)

        return pred_labels, pred_logits

    def _adversary_model(self, pred_logits, true_labels):
        """Compute the adversary predictions for the protected attribute.
        """
        with tf.variable_scope("adversary_model", reuse=tf.AUTO_REUSE):
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

        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            num_train_samples, self.features_dim = np.shape(features)

            # Setup placeholders
            self.features_ph = tf.placeholder(tf.float32, shape=[None, self.features_dim])
            self.protected_attributes_ph = tf.placeholder(tf.float32, shape=[None, 1])
            self.true_labels_ph = tf.placeholder(tf.float32, shape=[None, 1])
            self.keep_prob = tf.placeholder(tf.float32)

            # Obtain classifier predictions and classifier loss
            self.pred_labels, pred_logits = self._classifier_model_LR(self.features_ph, self.features_dim, self.keep_prob)
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


try:
    opts, args = getopt.getopt(sys.argv[1:], "h",
                               ["label_column=", "protect_columns=", "num_epochs=", "id=",
                                "num_proxies=", "verbose=", "balance=", "debias=", "batch_size="])
except getopt.GetoptError:
    print("Wrong format, see -h for help ...")
    print(sys.argv)
    sys.exit(2)

# INPUT PARAMS
LABEL_COL, PROTECT_COLS, NUM_EPOCH, ID, NUM_PROXIES, FILE_PATH, VERBOSE, \
BATCH_SIZE, BALANCE, DEBIAS = "income", ["gender"], 50, -1, 0, "../Datasets/adult_dataset/processed_adult.csv", 1, \
                              128, 1, 1

for opt, arg in opts:
    if opt == '-h':
        print(
            "--label_column=<label_column> \n "
            "--protect_columns=<protect_columns> (separated by a comma, no space) \n"
            "gender - male vs female (protected) \n"
            "race_White - white vs non-white (protected) \n"
            "This model can only handle one protected attribute"
            "--num_epoch=<num_epoch> \n--id=<id> \n"
            "--num_proxies= <num_proxies> \n--file_path=<file_path> \n--verbose=<verbose> \n-\n "
            "--batch_size=<batch_size> \n"
            "--balance=<balance> \n"
            "0: The training set and test set is not rebalanced in any way \n"
            "1: The training set is rebalanced in terms of labels and the test set is rebalanced in terms of label and"
            "groups/subgroups \n"
            "--debias=<debias>"
            "Whether to perform adversarial training to debias the model")
        sys.exit()
    if opt == '--label_column':
        LABEL_COL = int(arg)
    if opt == '--protect_columns':
        PROTECT_COLS = str(arg).split(",")
    if opt == '--num_epochs':
        NUM_EPOCH = int(arg)
    if opt == '--id':
        ID = int(arg)
    if opt == '--num_proxies':
        NUM_PROXIES = int(arg)
    if opt == '--file_path':
        FILE_PATH = int(arg)
    if opt == '--verbose':
        VERBOSE = int(arg)
    if opt == '--batch_size':
        BATCH_SIZE = int(arg)
    if opt == '--balance':
        BALANCE = int(arg)
    if opt == '--debias':
        DEBIAS = int(arg)

if len(PROTECT_COLS) >= 2:
    print("Arguments not valid: see flag -h for more information")
    sys.exit(1)

print(
    f"RUNNING SCRIPT WITH ARGUMENTS : -label_column={LABEL_COL} -protect_columns={PROTECT_COLS} -num_epoch={NUM_EPOCH} "
    f"-id={ID} -num_proxies={NUM_PROXIES}"
    f"-file_path={FILE_PATH} -verbose={VERBOSE} -batch_size={BATCH_SIZE} -balance={BALANCE}")

balanced = {"train_label_only": True, "test_label_only": False, "downsample": True} if BALANCE else None
train_dataset, test_dataset, _ = train_test_dataset(FILE_PATH,
                                                    LABEL_COL,
                                                    PROTECT_COLS,
                                                    is_scaled=True,
                                                    num_proxy_to_remove=NUM_PROXIES,
                                                    balanced=balanced
                                                    )
train_features, train_labels, train_protect = train_dataset.features, train_dataset.label, train_dataset.protect
test_features, test_labels, test_protect = test_dataset.features, test_dataset.label, test_dataset.protect

print("---------- MAPPING ----------")
print("Train: ", train_dataset.mapping)
print("Test: ", test_dataset.mapping)
print("-----------------------------")

hyperparameters = {'adversary_loss_weight': 0.1,
                   'batch_size': BATCH_SIZE,
                   'num_epochs': NUM_EPOCH,
                   }

train_accuracies, test_accuracies, fairness_accs, fairness_diffs = [], [], [], []

with tf.Session() as sess:
    model = AdversarialLogisticModel("training", sess, hyperparameters, debias=DEBIAS)
    trained_model = model.fit(train_features, train_labels, train_protect)
    train_pred_labels = trained_model.predict(train_features, train_labels, train_protect)
    test_pred_labels = trained_model.predict(test_features, test_labels, test_protect)

accs = equalizing_odds(test_pred_labels, test_labels, test_protect)
train_accuracies.append(accuracy_score(train_labels, train_pred_labels))
test_accuracies.append(accuracy_score(test_labels, test_pred_labels))
fairness_accs.append(accs)
fairness_diffs.append([max(acc) - min(acc) for acc in accs])

train_accuracies = np.array(train_accuracies)
test_accuracies = np.array(test_accuracies)
fairness_accs = np.array(fairness_accs)

print(f"Mapping: {test_dataset.mapping} \n")
print(f"Training accuracy: {train_accuracies.mean():3f} += {train_accuracies.std():3f}")
print(f"Test accuracy: {test_accuracies.mean():3f} += {test_accuracies.std():3f}")
print(f"Fairness accuracy: \n {np.mean(fairness_accs, axis=0)} += {np.std(fairness_accs, axis=0)}")
print(f"Fairness accuracy: \n {fairness_diffs}")

if ID >= 0:
    PATH = f"adversarial/checkpoints/model_ep_{NUM_EPOCH}/Debias_{DEBIAS}/Run_{ID}/stats.txt"
    os.makedirs(PATH, exist_ok=True)
    file = open(PATH + "/stats.txt", "w")
    file.write(f"Mapping: {test_dataset.mapping} \n")
    file.write(f"Training accuracy: {train_accuracies.mean():3f} += {train_accuracies.std():3f} \n")
    file.write(f"Test accuracy: {test_accuracies.mean():3f} += {test_accuracies.std():3f} \n")
    file.write(f"Fairness accuracy: \n {np.mean(fairness_accs, axis=0)} \n += \n {np.std(fairness_accs, axis=0)}")
    file.write(f"Fairness accuracy: \n {fairness_diffs}")
    file.close()
