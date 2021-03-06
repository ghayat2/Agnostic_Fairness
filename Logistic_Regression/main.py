import getopt
import sys
import os
from fairness_metrics import *
from load_dataset import *
from logistic_regression_model import *

try:
    opts, args = getopt.getopt(sys.argv[1:], "h",
                               ["label_column=", "protect_columns=", "mode=", "start_epoch=", "num_epochs=", "id=",
                                "num_trials=", "num_proxies=", "verbose=", "lr=", "update=", "weights_init=",
                                "update_lr=", "batch_size=", "balance="])
except getopt.GetoptError:
    print("Wrong format, see -h for help ...")
    print(sys.argv)
    sys.exit(2)

# INPUT PARAMS
LABEL_COL, PROTECT_COLS, MODE, START_EPOCH, NUM_EPOCH, ID, NUM_TRIALS, NUM_PROXIES, FILE_PATH, VERBOSE, \
LR_RATE, UPDATE, WEIGHTS_INIT, UPDATE_LR, BATCH_SIZE, BALANCE = "income", ["gender"], 0, 0, 40, -1, 1, 0, \
                                                                "../Datasets/adult_dataset/processed_adult.csv", 1, \
                                                                0.001, "cluster", 0, 10, 1000, 1

for opt, arg in opts:
    if opt == '-h':
        print(
            "--label_column=<label_column> \n "
            "--protect_columns=<protect_columns> (separated by a comma, no space) \n"
            "gender - male vs female (protected) \n"
            "race_White - white vs non-white (protected) \n"
            " --mode=<mode> \n"
            "0: Model is trained on bias dataset as it is, no reweighting \n"
            "1: Model is trained on customed dataset, where each sample is reweighted as to have the same number of \n"
            "minority and majority samples per class (only works when there is one protected column)"
            "2: Model is trained on customed dataset, where weights of each cluster is dynamically updated \n"
            "--update=<update> \n"
            "This parameter is only relevant when in MODE 2 \n"
            "cluster: each cluster has a weight \n"
            "sample: each sample has a weight \n"
            "--weights_init \n"
            "This parameter is only relevant in MODE 2 \n"
            "0: the cluster/sample weights are initialized with unit weight \n"
            "1: the cluster/sample weights are initialized with weights from MODE 1 (only works when there"
            "is one protected column) \n"
            "--start_epoch=<start_epoch> \n--num_epoch=<num_epoch> \n--id=<id> \n--num_trials=<num_trials> \n"
            "--num_proxies= <num_proxies> \n--file_path=<file_path> \n--verbose=<verbose> \n--lr=<lr> \n "
            "--update_lr=<update_lr> \n--batch_size=<batch_size> \n"
            "--balance=<balance> \n"
            "0: The training set and test set is not rebalanced in any way \n"
            "1: The training set is rebalanced in terms of labels and the test set is rebalanced in terms of label and"
            "groups/subgroups \n")
        sys.exit()
    if opt == '--label_column':
        LABEL_COL = int(arg)
    if opt == '--protect_columns':
        PROTECT_COLS = str(arg).split(",")
    if opt == '--mode':
        MODE = int(arg)
    if opt == '--start_epoch':
        START_EPOCH = int(arg)
    if opt == '--num_epochs':
        NUM_EPOCH = int(arg)
    if opt == '--id':
        ID = int(arg)
    if opt == '--num_trials':
        NUM_TRIALS = int(arg)
    if opt == '--num_proxies':
        NUM_PROXIES = int(arg)
    if opt == '--file_path':
        FILE_PATH = int(arg)
    if opt == '--verbose':
        VERBOSE = int(arg)
    if opt == '--lr':
        LR_RATE = float(arg)
    if opt == '--update':
        UPDATE = str(arg)
    if opt == '--weights_init':
        WEIGHTS_INIT = int(arg)
    if opt == '--update_lr':
        UPDATE_LR = float(arg)
    if opt == '--batch_size':
        BATCH_SIZE = int(arg)
    if opt == '--balance':
        BALANCE = int(arg)

if MODE not in [0, 1, 2] or (MODE == 1 and len(PROTECT_COLS) >= 2) or UPDATE not in ["cluster", "sample"] \
        or (WEIGHTS_INIT and len(PROTECT_COLS) >= 2):
    print("Arguments not valid: see flag -h for more information")
    sys.exit(1)

print(
    f"RUNNING SCRIPT WITH ARGUMENTS : -label_column={LABEL_COL} -protect_columns={PROTECT_COLS} -mode={MODE}"
    f" -start_epoch={START_EPOCH} -num_epoch={NUM_EPOCH} -id={ID} -num_trials={NUM_TRIALS} -num_proxies={NUM_PROXIES}"
    f"-file_path={FILE_PATH} -verbose={VERBOSE} -lr={LR_RATE} -update={UPDATE} -weights_init={WEIGHTS_INIT} "
    f"-update_lr={UPDATE_LR} -batch_size={BATCH_SIZE} -balance={BALANCE}")

"""
In order to test Case_1 and Case_2, we want the train and test set to have balanced labels, but we want the test set
 to also be balanced in terms of the sensitive attribute, while the train set should be bias.
"""
balanced = {"train_label_only": True, "test_label_only": False, "downsample": True} if BALANCE else None
PROTECT_VALUE = (0.0,) # Hard coded, corresponds to female for gender as a sensitive attribute and non-white for race as
                       # a sensitive attribute
train_dataset, test_dataset, train_w_minority = train_test_dataset(FILE_PATH, LABEL_COL, PROTECT_COLS,
                                                                   is_scaled=True,
                                                                   num_proxy_to_remove=NUM_PROXIES,
                                                                   balanced=balanced,
                                                                   reweighting=MODE,
                                                                   init=WEIGHTS_INIT)

print(f"Using weights {train_w_minority}")
print("---------- MAPPING ----------")
print("Train: ", train_dataset.mapping)
print("Test: ", test_dataset.mapping)
print("-----------------------------")

device = torch.device("cpu")
num_predictor_features = train_dataset[0][0].shape[0]
# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_accuracies, test_accuracies, fairness_accs, fairness_diffs = [], [], [], []
for trial in range(NUM_TRIALS):
    print(f"Trial {trial}: Training...")
    # Model and Optimizer
    predictor = Predictor(num_predictor_features).to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=LR_RATE)

    if MODE == 2:
        weights = [[1.0 for _ in range(2 ** len(PROTECT_COLS))] for _ in range(2)]
        weights[0][train_dataset.mapping[PROTECT_VALUE]] = train_w_minority[0]
        weights[1][train_dataset.mapping[PROTECT_VALUE]] = train_w_minority[1]

        args = [predictor, device, train_loader, optimizer, NUM_EPOCH, VERBOSE, 2 ** len(PROTECT_COLS),
                2, UPDATE_LR, weights if WEIGHTS_INIT else None]
        func = train_sample_reweight if UPDATE == "sample" else train_cluster_reweight
    else:
        args = [predictor, device, train_loader, optimizer, NUM_EPOCH, VERBOSE,
                train_w_minority if MODE == 1 else None, train_dataset.mapping[PROTECT_VALUE]]
        func = train
    train_history = func(*args)

    # print(train_history[[c for c in train_history.columns if "cluster" in c]])

    ###### Test set
    train_pred_labels, train_loss, train_accuracy, _ = test(predictor, device, train_loader)
    test_pred_labels, test_loss, test_accuracy, _ = test(predictor, device, test_loader)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    accs = equalizing_odds(test_pred_labels, test_loader.dataset.label, test_loader.dataset.protect)
    fairness_accs.append(accs)
    fairness_diffs.append([max(acc) - min(acc) for acc in accs])

    #### Saving checkpoint
    if ID >= 0:
        PATH = f"Case_{MODE + 1}/checkpoints/model_ep_{START_EPOCH + NUM_EPOCH}/Run_{ID}/trial_{trial}"
        LOSS = "BCELoss"

        os.makedirs(PATH, exist_ok=True)
        torch.save({
            'epoch': START_EPOCH + NUM_EPOCH,
            'model_state_dict': predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
        }, PATH + "/checkpoint.pt ")
        train_history.to_pickle(PATH + "/train_history.pkl")

train_accuracies = np.array(train_accuracies)
test_accuracies = np.array(test_accuracies)
fairness_accs = np.array(fairness_accs)

print(f"Mapping: {test_dataset.mapping} \n")
print(f"Training accuracy: {train_accuracies.mean():3f} += {train_accuracies.std():3f}")
print(f"Test accuracy: {test_accuracies.mean():3f} += {test_accuracies.std():3f}")
print(f"Fairness accuracy: \n {np.mean(fairness_accs, axis=0)} += {np.std(fairness_accs, axis=0)}")
print(f"Fairness diffs: \n {fairness_diffs}")

if ID >= 0:
    PATH = f"Case_{MODE + 1}/checkpoints/model_ep_{START_EPOCH + NUM_EPOCH}/Run_{ID}/stats.txt"
    file = open(PATH, "w")
    file.write(f"Mapping: {test_dataset.mapping} \n")
    file.write(f"Training accuracy: {train_accuracies.mean():3f} += {train_accuracies.std():3f} \n")
    file.write(f"Test accuracy: {test_accuracies.mean():3f} += {test_accuracies.std():3f} \n")
    file.write(f"Fairness accuracy: \n {np.mean(fairness_accs, axis=0)} \n += \n {np.std(fairness_accs, axis=0)}")
    file.write(f"Fairness difference: \n {fairness_diffs}")
    file.close()
