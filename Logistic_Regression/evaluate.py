import getopt
import sys
import os
from fairness_metrics import *
from load_dataset import *
from logistic_regression_model import *


try:
    opts, args = getopt.getopt(sys.argv[1:], "h",
                               ["label_column=", "protect_columns=", "mode=", "start_epoch=", "num_epochs=", "id=",
                                "num_trials=", "num_proxies=", "verbose=", "lr=", "cluster_lr=", "batch_size="])
except getopt.GetoptError:
    print("Wrong format ...")
    print(sys.argv)
    sys.exit(2)

# INPUT PARAMS
data_dir = '../Datasets/doctor_nurse/train_test_split'
LABEL_COL, PROTECT_COLS, MODE, START_EPOCH, NUM_EPOCH, ID, NUM_TRIALS, NUM_PROXIES, FILE_PATH, VERBOSE, \
LR_RATE, CLUSTER_LR, BATCH_SIZE = "income", ["gender"], 0, 0, 40, 1, False, 0, \
                                  "../Datasets/adult_dataset/processed_adult.csv", 1, 0.001, 10, 1000

for opt, arg in opts:
    if opt == '-h':
        print(
            "main.py  --label_column=<label_column> --protect_columns=<protect_columns (separated by a comma, no space)>"
            " --mode=<mode>"
            "0: Model is trained on bias dataset as it is, no reweighting"
            "1: Model is trained on customed dataset, where each sample is reweighted as to have same number of"
            "protected samples per class (only work when there is one protected column)"
            "2: Model is traied on customed dataset, where weights of each cluster is dynamically reweighted"
            "--start_epoch=<start_epoch> --num_epoch=<num_epoch> --id=<id> --num_trials=<num_trials>"
            "--num_proxies= <num_proxies> --file_path=<file_path> --verbose=<verbose> --lr=<lr> "
            "--cluster_lr=<cluster_lr> --batch_size=<batch_size>")
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
    if opt == '--cluster_lr':
        CLUSTER_LR = float(arg)
    if opt == '--batch_size':
        BATCH_SIZE = int(arg)

if MODE not in [0, 1, 2] or MODE == 1 and len(PROTECT_COLS) >= 2:
    print("Arguments not valid: see flag -h for more information")
    sys.exit(1)

print(
    f"RUNNING SCRIPT WITH ARGUMENTS : -label_column={LABEL_COL} -protect_columns={PROTECT_COLS} -mode={MODE}"
    f" -start_epoch={START_EPOCH} -num_epoch={NUM_EPOCH} -id={ID} -num_trials={NUM_TRIALS} -num_proxies={NUM_PROXIES}"
    f"-file_path={FILE_PATH} -verbose={VERBOSE} -lr={LR_RATE} -cluster_lr={CLUSTER_LR} -batch_size={BATCH_SIZE}")

"""
In order to test Case_1 and Case_2, we want the train and test set to have balanced labels, but we want the test set
 to also be balanced in terms of the sensitive attribute, while the train set should be bias.
"""
balanced = {"train_label_only": True, "test_label_only": False, "downsample": True}
train_dataset, test_dataset, train_w_minority = train_test_dataset(FILE_PATH, LABEL_COL, PROTECT_COLS,
                                                                   is_scaled=True,
                                                                   num_proxy_to_remove=NUM_PROXIES,
                                                                   balanced=balanced,
                                                                   reweighting=MODE)
device = torch.device("cpu")
num_predictor_features = train_dataset[0][0].shape[0]
# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_accuracies, test_accuracies, fairness_accs, fairness_diffs = [], [], [], []
for trial in range(NUM_TRIALS):
    print(f"Trial {trial}: Evaluating...")
    # Model and Optimizer
    predictor = Predictor(num_predictor_features).to(device)

    print(os.getcwd())
    PATH = f"/Case_{MODE + 1}/checkpoints/model_ep_{START_EPOCH + NUM_EPOCH}/Run_{ID}/trial_{trial}/checkpoint.pt "
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    predictor.load_state_dict(checkpoint["model_state_dict"])

    ###### Test set
    train_pred_labels, train_loss, train_accuracy = test(predictor, device, train_loader)
    test_pred_labels, test_loss, test_accuracy = test(predictor, device, test_loader)

    train_accuracies.append(train_accuracy);
    test_accuracies.append(test_accuracy)

    accs = equalizing_odds(test_pred_labels, test_loader.dataset.label, test_loader.dataset.protect)
    fairness_accs.append(accs)
    fairness_diffs.append([max(acc) - min(acc) for acc in accs])

train_accuracies = np.array(train_accuracies)
test_accuracies = np.array(test_accuracies)
fairness_accs = np.array(fairness_accs)

print(f"Training accuracy: {train_accuracies.mean():3f} += {train_accuracies.std():3f}")
print(f"Test accuracy: {test_accuracies.mean():3f} += {test_accuracies.std():3f}")
print(f"Fairness accuracy: \n {np.mean(fairness_accs, axis=0)} += {np.std(fairness_accs, axis=0)}")
print(f"Fairness accuracy: \n {fairness_diffs}")
