import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
import tensorflow as tf
import time
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import sem
import train_clp_hire as SenSR
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.decomposition import TruncatedSVD
import AdvDebCustom

def get_hire_data():
    '''
    Preprocess the hire data set by removing some features and put hire data into a BinaryLabelDataset
    '''

    headers = ['Age', 'Accessibility', 'EdLevel', 'Employment', 'Gender',
               'MentalHealth', 'MainBranch', 'YearsCode', 'YearsCodePro', 'Country',
               'PreviousSalary', 'HaveWorkedWith', 'ComputerSkills', 'Employed']
    train = pd.read_csv('hire_traing_data.csv', header = None)
    test = pd.read_csv('hire_testing_data.csv', header = None)
    df = pd.concat([train, test], ignore_index=True)
    df.columns = headers

    df = pd.get_dummies(df, columns=[headers[0], headers[2], headers[4], headers[6], headers[9], headers[11], headers[12]])

    delete_these += ['Country_ Cambodia', 'Country_ Canada', 'Country_ China', 'Country_ Columbia', 'Country_ Cuba', 'Country_ Dominican-Republic', 'Country_ Ecuador', 'Country_ El-Salvador', 'Country_ England', 'Country_ France', 'Country_ Germany', 'Country_ Greece', 'Country_ Guatemala', 'Country_ Haiti', 'Country_ Holand-Netherlands', 'Country_ Honduras', 'Country_ Hong', 'Country_ Hungary', 'Country_ India', 'Country_ Iran', 'Country_ Ireland', 'Country_ Italy', 'Country_ Jamaica', 'Country_ Japan', 'Country_ Laos', 'Country_ Mexico', 'Country_ Nicaragua', 'Country_ Outlying-US(Guam-USVI-etc)', 'Country_ Peru', 'Country_ Philippines', 'Country_ Poland', 'Country_ Portugal', 'Country_ Puerto-Rico', 'Country_ Scotland', 'Country_ South', 'Country_ Taiwan', 'Country_ Thailand', 'Country_ Trinadad&Tobago', 'Country_ United-States', 'Country_ Vietnam', 'Country_ Yugoslavia']
    delete_these += ['MentalHealth', 'Accessibility']

    df.drop(delete_these, axis=1, inplace=True)

    return BinaryLabelDataset(df = df, label_names = ['Employed'], protected_attribute_names = ['Gender_ Man'])

def preprocess_hire_data(seed = 0):
    '''
    Description: Ths code (1) standardizes the continuous features, (2) one hot encodes the categorical features, (3) splits into a train (80%) and test set (20%), (4) based on this data, create another copy where gender is deleted as a predictive feature and the feature we predict is gender (used by SenSR when learning the sensitive directions)

    Input: seed: the seed used to split data into train/test
    '''
    # Get the dataset and split into train and test
    dataset_orig = get_hire_data()

    # we will standardize continous features
    continous_features = ['YearsCode', 'YearsCodePro', 'PreviousSalary']
    continous_features_indices = [dataset_orig.feature_names.index(feat) for feat in continous_features]

    # get a 80%/20% train/test split
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=True, seed = seed)
    SS = StandardScaler().fit(dataset_orig_train.features[:, continous_features_indices])
    dataset_orig_train.features[:, continous_features_indices] = SS.transform(dataset_orig_train.features[:, continous_features_indices])
    dataset_orig_test.features[:, continous_features_indices] = SS.transform(dataset_orig_test.features[:, continous_features_indices])

    X_train = dataset_orig_train.features
    X_test = dataset_orig_test.features

    y_train = dataset_orig_train.labels
    y_test = dataset_orig_test.labels

    one_hot = OneHotEncoder(sparse_output=False)
    one_hot.fit(y_train.reshape(-1,1))
    names_income = one_hot.categories_
    y_train = one_hot.transform(y_train.reshape(-1,1))
    y_test = one_hot.transform(y_test.reshape(-1,1))

    # Also create a train/test set where the predictive features (X) do not include gender and gender is what you want to predict (y). This is used when learnng the sensitive direction for SenSR
    X_gender_train = np.delete(X_train, [dataset_orig_test.feature_names.index(feat) for feat in ['Gender_ Man']], axis = 1)
    X_gender_test = np.delete(X_test, [dataset_orig_test.feature_names.index(feat) for feat in ['Gender_ Man']], axis = 1)

    y_gender_train = dataset_orig_train.features[:, dataset_orig_train.feature_names.index('Gender_ Man')]
    y_gender_test = dataset_orig_test.features[:, dataset_orig_test.feature_names.index('Gender_ Man')]

    one_hot.fit(y_gender_train.reshape(-1,1))
    names_gender = one_hot.categories_
    y_gender_train = one_hot.transform(y_gender_train.reshape(-1,1))
    y_gender_test = one_hot.transform(y_gender_test.reshape(-1,1))

    return X_train, X_test, y_train, y_test, X_gender_train, X_gender_test, y_gender_train, y_gender_test, dataset_orig_train, dataset_orig_test
    # names_income, names_gender

def save_to_file(directory, variable, name):
    timestamp = str(int(time.time()))
    with open(directory + name + '_' + timestamp + '.txt', "w") as f:
        f.write(str(np.mean(variable))+"\n")
        f.write(str(sem(variable))+"\n")
        for s in variable:
            f.write(str(s) +"\n")

def compute_gap_RMS_and_gap_max(data_set):
    '''
    Description: computes the gap RMS and max gap
    Input: data_set: a BinaryLabelDataset (from the aif360 module)
    '''
    TPR = -1*data_set.false_negative_rate_difference()
    TNR = -1*data_set.false_positive_rate_difference()

    return np.sqrt(1/2*(TPR**2 + TNR**2)), max(np.abs(TPR), np.abs(TNR))

def compute_balanced_accuracy(data_set):
    '''
    Description: computes the balanced accuracy, i.e. the average of TPR and TNR
    Input: data_set: a BinaryLabelDataset (from the aif360 module)
    '''
    TPR = data_set.true_positive_rate()
    TNR = data_set.true_negative_rate()
    return 0.5*(TPR+TNR)

def get_consistency(X, weights=0, proj = 0, gender_idx = 39, man_idx = 33, woman_idx = 38, adv = 0, dataset_orig_test = 0):
    '''
    Description: Ths function computes gender consistency.
    Input:
        X: numpy matrix of predictive features
        weights: learned weights for project, baseline, and sensr
        proj: if using the project first baseline, this is the projection matrix
        gender_idx: column corresponding to the binary gender variable
        man_idx: column corresponding to the man variable
        woman_idx: column corresponding to the woman variable
        adv: the adversarial debiasing object if using adversarial Adversarial Debiasing
        dataset_orig_test: this is the data in a BinaryLabelDataset format when using adversarial debiasing
    '''

    if adv == 0:
        N, D = X.shape
        K = 1

        tf_X = tf.placeholder(tf.float32, shape=[None,D])
        tf_y = tf.placeholder(tf.float32, shape=[None,K], name='response')

        n_units = weights[1].shape
        n_units = n_units[0]

        _, l_pred, _, _ = SenSR.forward(tf_X, tf_y, weights=weights, n_units = n_units, activ_f = tf.nn.relu)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        n, _ = X.shape
        X_men = np.copy(X)
        X_men[:,gender_idx] = 0
        X_men[:,man_idx] = 1

        if np.ndim(proj) != 0:
            X_men = X_men@proj

        if adv == 0:
            man_logits = l_pred.eval(feed_dict={tf_X: X_men})
            man_preds = np.argmax(man_logits, axis = 1)
        else:
            dataset_mod = dataset_orig_test.copy(deepcopy=True)
            dataset_mod.features = X_men
            dataset_mod, _ = adv.predict(dataset_mod)
            man_preds = [i[0] for i in dataset_mod.labels]

        X_women = np.copy(X)
        X_women[:,gender_idx] = 0
        X_women[:,woman_idx] = 1

        if np.ndim(proj) != 0:
            X_women = X_women@proj

        if adv == 0:
            women_logits = l_pred.eval(feed_dict={tf_X: X_women})
            women_preds = np.argmax(women_logits, axis = 1)
        else:
            dataset_mod = dataset_orig_test.copy(deepcopy=True)
            dataset_mod.features = X_women
            dataset_mod, _ = adv.predict(dataset_mod)
            women_preds = [i[0] for i in dataset_mod.labels]

        gender_consistency = np.mean([1 if man_preds[i] == women_preds[i] else 0 for i in range(len(man_preds))])

        return gender_consistency
        
def get_sensitive_directions_and_projection_matrix(X_gender_train, y_gender_train, X_gender_test, y_gender_test, gender_idx = 39 ):
    '''
    Description: Get the sensitive directions and projection matrix. The sensitive directions include the gender direction as well as the learned hyperplane that predicts gender (without using gender as a predictive feature of course).
    '''
    weights, train_logits, test_logits = SenSR.train_nn(X_gender_train, y_gender_train, X_test = X_gender_test, y_test = y_gender_test, n_units=[], l2_reg=.1, batch_size=5000, epoch=5000, verbose=False)
    
#    all_mean = X_gender_train.mean(axis=0)
#    m0 = X_gender_train[y_gender_train[:,0]==1].mean(axis=0) - all_mean
#    m1 = X_gender_train[y_gender_train[:,1]==1].mean(axis=0) - all_mean
    
#    weights = np.vstack((m0,m1)).T
    
    n, d = weights[0].shape
    print(n,d)
    sensitive_directions = []

    # transform the n-dimensional weights back into the full n+1 dimensional space where the gender coordinate is zeroed out
    full_weights = np.zeros((n+1,d))

    #before the gender coordinate, the coordinates of the full_weights and learned weights correspond to the same features
    for i in range(gender_idx):
        full_weights[i,:] = weights[0][i,:]

    #after the gender coordinate, the i-th coordinate of the full_weights correspond to the (i-1)-st coordinate of the learned weights
    for i in range(gender_idx+1, n+1):
        full_weights[i, :] = weights[0][i-1,:]

    sensitive_directions.append(full_weights.T)

    
    temp_direction = np.zeros((n+1,1)).reshape(1,-1)
    temp_direction[0, gender_idx] = 1
    sensitive_directions.append(np.copy(temp_direction))

    sensitive_directions = np.vstack(sensitive_directions)

    return sensitive_directions, SenSR.compl_svd_projector(sensitive_directions)

def get_metrics(dataset_orig, preds, verbose=True):
    '''
    Description: This code computes accuracy, balanced accuracy, max gap and gap rms for gender
    Input: dataset_orig: a BinaryLabelDataset (from the aif360 module)
            preds: predictions
    '''
    dataset_learned_model = dataset_orig.copy()
    dataset_learned_model.labels = preds

    # wrt gender
    privileged_groups = [{'Gender_ Man': 1}]
    unprivileged_groups = [{'Gender_ Man': 0}]

    classified_metric = ClassificationMetric(dataset_orig,
                                                     dataset_learned_model,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

    bal_acc = compute_balanced_accuracy(classified_metric)

    gender_gap_rms, gender_max_gap = compute_gap_RMS_and_gap_max(classified_metric)
    if verbose:
        print("Test set: gender gap rms = %f" % gender_gap_rms)
        print("Test set: gender max gap rms = %f" % gender_max_gap)
        print("Test set: Balanced TPR = %f" % bal_acc)


    return classified_metric.accuracy(), bal_acc, gender_gap_rms, gender_max_gap

def run_baseline_experiment(X_train, y_train, X_test, y_test):
    return SenSR.train_nn(X_train, y_train, X_test = X_test, y_test = y_test, n_units=[100], l2_reg=0., lr = .00001, batch_size=1000, epoch=60000, verbose=False)

    
def run_SenSR_experiment(X_train, X_test, X_gender_train, X_gender_test, y_train, y_test, y_gender_train, y_gender_test, directory, i):
    sensitive_directions, _ = get_sensitive_directions_and_projection_matrix(X_gender_train, y_gender_train, X_gender_test, y_gender_test)
    return SenSR.train_fair_nn(
        X_train,
        y_train,
        sensitive_directions,
        X_test=X_test,
        y_test=y_test,
        n_units = [100],
        lr=.0001,
        batch_size=1000,
        epoch=12000,
        verbose=True,
        l2_reg=0.,
        lamb_init=2.,
        subspace_epoch=50,
        subspace_step=10,
        eps=.001,
        full_step=.0001,
        full_epoch=40)

def run_project_experiment(X_train, X_test, X_gender_train, X_gender_test, y_train, y_test, y_gender_train, y_gender_test, directory, i):
    _, proj_compl = get_sensitive_directions_and_projection_matrix(X_gender_train, y_gender_train, X_gender_test, y_gender_test)

    np.save(directory+'proj_compl_'+str(i), proj_compl)

    X_train_proj = X_train@proj_compl
    X_test_proj = X_test@proj_compl
    weights, train_logits, test_logits = SenSR.train_nn(
        X_train_proj,
        y_train,
        X_test = X_test_proj,
        y_test = y_test,
        n_units=[100],
        l2_reg=0.,
        lr = .00001,
        batch_size=1000,
        epoch=60000,
        verbose=False)
    return weights, train_logits, test_logits, proj_compl

def run_adv_deb_experiment(dataset_orig_train, dataset_orig_test):
    sess = tf.Session()
    tf.name_scope("my_scope")
    privileged_groups = [{'Gender_ Man': 1}]
    unprivileged_groups = [{'Gender_ Man': 0}]
    adv = AdvDebCustom.AdversarialDebiasing(unprivileged_groups, privileged_groups, "my_scope", sess, seed=None, adversary_loss_weight=0.001, num_epochs=500, batch_size=1000, classifier_num_hidden_units=100, debias=True)

    _ = adv.fit(dataset_orig_train)
    test_data, _ = adv.predict(dataset_orig_test)

    return adv, test_data.labels

def run_experiments(name, num_exp, directory):
    '''
    Description: Run each experiment num_exp times where a new train/test split is generated. Save results in the path specified by directory

    Inputs: name: name of the experiment. Valid choices are baseline, project, SenSR, adv_deb
    '''

    if name not in ['baseline', 'project', 'SenSR', 'adv_deb']:
        raise ValueError('You did not specify a valid experiment to run.')

    gender_consistencies = []

    accuracy = []
    balanced_accuracy = []

    gender_gap_rms = []
    gender_max_gap = []
    
    exp_seeds = [1345]    

    for i in exp_seeds:
        print('On experiment', i)

        # get train/test data
        X_train, X_test, y_train, y_test, X_gender_train, X_gender_test, y_gender_train, y_gender_test, dataset_orig_train, dataset_orig_test = preprocess_hire_data(seed = i)

        tf.reset_default_graph()

        # run experiments
        if name == 'baseline':
            weights, train_logits, test_logits  = run_baseline_experiment(X_train, y_train, X_test, y_test)
        elif name == 'SenSR':
            weights, train_logits, test_logits = run_SenSR_experiment(X_train, X_test, X_gender_train, X_gender_test, y_train, y_test, y_gender_train, y_gender_test, directory, i)
        elif name == 'project':
            weights, train_logits, test_logits, proj_compl = run_project_experiment(X_train, X_test, X_gender_train, X_gender_test, y_train, y_test, y_gender_train, y_gender_test, directory, i)
        elif name == 'adv_deb':
            adv, preds = run_adv_deb_experiment(dataset_orig_train, dataset_orig_test)

        # get consistency

        if name == 'project':
            gender_consistency = get_consistency(X_test, weights = weights, proj = proj_compl)
        elif name == 'adv_deb':
            gender_consistency = get_consistency(X_test, adv = adv, dataset_orig_test = dataset_orig_test)
        else:
            gender_consistency = get_consistency(X_test, weights = weights)
        gender_consistencies.append(gender_consistency)
        print('gender consistency', gender_consistency)

        # get accuracy, balanced accuracy, gender gap rms, gender max gap

        if name != 'adv_deb':
            np.save(directory+'weights_'+str(i), weights)
            preds = np.argmax(test_logits, axis = 1)
        acc_temp, bal_acc_temp, gender_gap_rms_temp, gender_max_gap_temp = get_metrics(dataset_orig_test, preds)

        gender_gap_rms.append(gender_gap_rms_temp)
        gender_max_gap.append(gender_max_gap_temp)
        accuracy.append(acc_temp)
        balanced_accuracy.append(bal_acc_temp)

    # save results to file
    save_info = [
        (gender_consistencies, 'gender_consistencies'),
        (accuracy, 'accuracy'),
        (balanced_accuracy, 'balanced_accuracy'),
        (gender_gap_rms, 'gender_gap_rms'),
        (gender_max_gap, 'gender_max_gap')]

    for (variable, name) in save_info:
        save_to_file(directory, variable, name)

num_exp = 1
#change the directory below in run_experiments to the directoy where you want to save the results of the experiments
run_experiments('baseline', num_exp, '/Users/jamin/Harvard_CS/cs226r/CS-226-Final-Project/hiring_data_experiment/experiments')
run_experiments('project', num_exp, '/Users/jamin/Harvard_CS/cs226r/CS-226-Final-Project/hiring_data_experiment/experiments')
t_s = time.time()
run_experiments('SenSR', num_exp, '/Users/jamin/Harvard_CS/cs226r/CS-226-Final-Project/hiring_data_experiment/experiments')
t_e = time.time()
print('Took', (t_e - t_s)/60, 'min')
run_experiments('adv_deb', num_exp, '/Users/jamin/Harvard_CS/cs226r/CS-226-Final-Project/hiring_data_experiment/experiments')

X_train, X_test, y_train, y_test, X_gender_train, X_gender_test, y_gender_train, y_gender_test, dataset_orig_train, dataset_orig_test = preprocess_hire_data(seed=0)

xx, tt = get_sensitive_directions_and_projection_matrix(X_gender_train, y_gender_train, X_gender_test, y_gender_test, gender_idx = 39 )
