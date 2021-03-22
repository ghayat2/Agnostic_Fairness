import gzip
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import tensorflow.compat.v1 as tf


class AdversarialLogisticModel(object):
    """A model for doing adversarial training of logistic models."""
    def __init__(self,  
                scope_name,
                sess,
                hyperparameters,
                seed=None,
                debias=False):

        self.sess = sess
        self.scope_name = scope_name
        self.seed = seed

        self.adversary_loss_weight = hyperparameters['adversary_loss_weight']
        self.num_epochs = hyperparameters["num_epochs"]
        self.batch_size = hyperparameters["batch_size"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.debias = debias
       
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

    def adversarial_model(self, pred_logits, true_labels):
        # why do we need to have access to true labels when
        # guessing the protected attribute ?
        with tf.variable_scope("adversary_model"):
            c = tf.get_variable('c', initializer=tf.constant(1.0))
            s = tf.sigmoid((1 + tf.abs(c)) * pred_logits)

            W2 = tf.get_variable('W2', [3, 1], initializer=tf.glorot_uniform_initializer())
            b2 = tf.Variable(tf.zeros(shape=[1]), name='b2')

            pred_protected_attribute_logits = tf.matmul(tf.concat([s, s * true_labels, s * (1.0 - true_labels)], axis=1), W2) + b2
            pred_protected_attribute_labels = tf.sigmoid(pred_protected_attribute_logits)

        return pred_protected_attribute_labels, pred_protected_attribute_logits


    def fit(self, features, labels, protect):
        if self.seed is not None:
            np.random.seed(self.seed)

        with tf.variable_scope(self.scope_name):
            num_train_samples, self.features_dim = np.shape(features)

            # Setup placeholders
            self.features_ph = tf.placeholder(tf.float32, shape=[None, self.features_dim])
            self.protected_attributes_ph = tf.placeholder(tf.float32, shape=[None,1])
            self.true_labels_ph = tf.placeholder(tf.float32, shape=[None,1])

            # Obtain classifier predictions and classifier loss
            self.pred_labels, pred_logits = self.predictor_model(self.features_ph, self.features_dim)
            pred_labels_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_labels_ph, logits=pred_logits))

            if self.debias:
                # Obtain adversary predictions and adversary loss
                pred_protected_attributes_labels, pred_protected_attributes_logits = self.adversarial_model(pred_logits, self.true_labels_ph)
                pred_protected_attributes_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.protected_attributes_ph, logits=pred_protected_attributes_logits))

            # Setup optimizers with learning rates
            global_step = tf.Variable(0, trainable=False)
            # starter_learning_rate = 0.001
            # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
            predictor_opt = tf.train.AdamOptimizer(self.learning_rate)
            if self.debias:
                adversary_opt = tf.train.AdamOptimizer(self.learning_rate)

            predictor_vars = [var for var in tf.trainable_variables() if 'predictor_model' in var.name]
            if self.debias:
                adversary_vars = [var for var in tf.trainable_variables() if 'adversary_model' in var.name]
                # Compute gradient of adversarial loss with respect to predictor variables
                adversary_grads = {var: grad for (grad, var) in adversary_opt.compute_gradients(pred_protected_attributes_loss, var_list=predictor_vars)}
            normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

            predictor_grads = []
            for (grad,var) in predictor_opt.compute_gradients(pred_labels_loss, var_list=predictor_vars):
                if self.debias:
                    unit_adversary_grad = normalize(adversary_grads[var])
                    grad -= tf.reduce_sum(grad * unit_adversary_grad) * unit_adversary_grad
                    grad -= self.adversary_loss_weight * adversary_grads[var]
                predictor_grads.append((grad, var))
            # Update predictor parameters
            predictor_minimizer = predictor_opt.apply_gradients(predictor_grads, global_step=global_step)

            if self.debias:
                # Update adversary parameters
                adversary_minimizer = adversary_opt.minimize(pred_protected_attributes_loss, var_list=adversary_vars, global_step=global_step)

            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

            # Begin training
            for epoch in range(self.num_epochs):
                shuffled_ids = np.random.choice(num_train_samples, num_train_samples)
                for i in range(num_train_samples//self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size*i: self.batch_size*(i+1)]
                    batch_features = features[batch_ids]
                    batch_labels = np.reshape(labels[batch_ids], [-1,1])
                    batch_protected_attributes = np.reshape(protect[batch_ids], [-1,1])

                    batch_feed_dict = {self.features_ph: batch_features,
                                        self.true_labels_ph: batch_labels,
                                        self.protected_attributes_ph: batch_protected_attributes
                                        }
                    if self.debias:
                        _, _, pred_labels_loss_value, pred_protected_attributes_loss_vale = self.sess.run([predictor_minimizer, adversary_minimizer,
                                       pred_labels_loss, pred_protected_attributes_loss], feed_dict=batch_feed_dict)
                        if i % 200 == 0:
                            print("epoch %d; iter: %d; batch predictor loss: %f; batch adversarial loss: %f" % (epoch, i, pred_labels_loss_value,
                                                                                     pred_protected_attributes_loss_vale))
                    else:
                        _, pred_labels_loss_value = self.sess.run([predictor_minimizer, pred_labels_loss], feed_dict=batch_feed_dict)
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
            batch_labels = np.reshape(labels[batch_ids], [-1,1])
            batch_protected_attributes = np.reshape(protect[batch_ids], [-1,1])

            batch_feed_dict = {self.features_ph: batch_features,
                               self.true_labels_ph: batch_labels,
                               self.protected_attributes_ph: batch_protected_attributes,
                               }

            pred_labels += self.sess.run(self.pred_labels, feed_dict=batch_feed_dict)[:,0].tolist()
            samples_covered += len(batch_features)

        true_labels = list(labels.reshape(1, -1)[0])
        
        pred_labels = np.array([1 if i > 0.5 else 0 for i in pred_labels]).reshape(-1,1)
        
        return pred_labels

def get_data(filename):
    return pd.read_csv(filename) 

def split_train_test(compas_df, train=0.75):
    shuffled = np.random.RandomState(0).permutation(compas_df.index)
    n_train = int(len(shuffled) * train)
    i_train, i_test = shuffled[:n_train], shuffled[n_train:]
    return compas_df.loc[i_train], compas_df.loc[i_test]


def process_data(df, drop_columns, label_column, protect_column):
    df = df.drop(columns=drop_columns, axis=1)
    feature_columns = set(df.columns) - {label_column, protect_column}
    features = df[list(feature_columns)].values
    labels = df[label_column].values.reshape(-1,1)
    protect = df[protect_column].values.reshape(-1,1)
    return (features, labels, protect)


compas_df = get_data("compas_data.csv")
label_column = "is_recid"
protect_column = "race_African-American"

drop_columns = ['decile_score', 'sex_Male', 'race_Caucasian']
train_df, test_df = split_train_test(compas_df)
train_features, train_labels, train_protect = process_data(train_df, drop_columns, label_column, protect_column)
test_features, test_labels, test_protect = process_data(test_df, drop_columns, label_column, protect_column)

hyperparameters = {'adversary_loss_weight':0.01, 
                    'batch_size':64, 
                    'num_epochs':50, 
                    'learning_rate':0.01
                    }
with tf.Session() as sess:
    model = AdversarialLogisticModel("training", sess, hyperparameters, debias=True)
    trained_model = model.fit(train_features, train_labels, train_protect)
    train_pred_labels = trained_model.predict(train_features, train_labels, train_protect)
    test_pred_labels = trained_model.predict(test_features, test_labels, test_protect)


def binary_confusion_matrix(true_labels, pred_labels, protect, protect_group):
    indices = np.where(protect == protect_group)
    group_pred_labels = pred_labels[indices]
    group_true_labels = true_labels[indices]

    return confusion_matrix(group_true_labels, group_pred_labels)

black_confusion_matrix = binary_confusion_matrix(test_labels, test_pred_labels, test_protect, 1)
white_confusion_matrix = binary_confusion_matrix(test_labels, test_pred_labels, test_protect, 0)
print(black_confusion_matrix, " Blacks")
print(white_confusion_matrix, " Whites")

def false_positive_rate(group_confusion_matrix):
    return group_confusion_matrix[0][1]/np.sum(group_confusion_matrix[:,1])

def true_positive_rate(group_confusion_matrix):
    return group_confusion_matrix[1][1]/np.sum(group_confusion_matrix[1,:])

def false_negative_rate(group_confusion_matrix):
    return group_confusion_matrix[1][0]/np.sum(group_confusion_matrix[:,0])

def true_negative_rate(group_confusion_matrix):
    return group_confusion_matrix[0][0]/np.sum(group_confusion_matrix[0,:])

def false_positive_rate_difference(confusion_matrix_1, confusion_matrix_2):
    return false_positive_rate(confusion_matrix_1) - false_positive_rate(confusion_matrix_2) 

def true_positive_rate_difference(confusion_matrix_1, confusion_matrix_2):
    return true_positive_rate(confusion_matrix_1) - true_positive_rate(confusion_matrix_2) 

def false_negative_rate_difference():
    return false_negative_rate(confusion_matrix_1) - false_negative_rate(confusion_matrix_2)

def average_odds_difference(confusion_matrix_1, confusion_matrix_2):
    fpr_difference = false_positive_rate_difference(confusion_matrix_1, confusion_matrix_2)
    tpr_difference = true_positive_rate_difference(confusion_matrix_1, confusion_matrix_2)

    return 0.5*(fpr_difference + tpr_difference)

def statistical_parity_difference(confusion_matrix_1, confusion_matrix_2):

    frac_prediced_positive_1 = np.sum(confusion_matrix_1[:,1])/np.sum(confusion_matrix_1)
    frac_prediced_positive_2 = np.sum(confusion_matrix_2[:,1])/np.sum(confusion_matrix_2)

    return frac_prediced_positive_1 - frac_prediced_positive_2

black_fpr = false_positive_rate(black_confusion_matrix)
white_fpr = false_positive_rate(white_confusion_matrix)

print("Train Accuracy: ", accuracy_score(train_labels, train_pred_labels))
print("Test Accuracy: ", accuracy_score(test_labels, test_pred_labels))


print(black_fpr, white_fpr, " FPR blacks, whites")
print(statistical_parity_difference(black_confusion_matrix, white_confusion_matrix), " statistical_parity_difference")
print(average_odds_difference(black_confusion_matrix, white_confusion_matrix), " average_odds_difference")



