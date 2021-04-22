import getopt
import sys
import os
from fairness_metrics import *
from load_dataset import *
from logistic_regression_model import *

try:
    opts, args = getopt.getopt(sys.argv[1:], "h",
                               ["label_column=", "protect_columns=", "num_epochs=", "id=",
                                "num_trials=", "num_proxies=", "verbose=", "lr=",
                                "batch_size=", "balance=", "keep=", "filter_maj=", "filter_min="])
except getopt.GetoptError:
    print("Wrong format, see -h for help ...")
    print(sys.argv)
    sys.exit(2)

# INPUT PARAMS
LABEL_COL, PROTECT_COLS, NUM_EPOCH, ID, NUM_TRIALS, NUM_PROXIES, FILE_PATH, VERBOSE, \
LR_RATE, BATCH_SIZE, BALANCE, KEEP, FILTER_MAJ, FILTER_MIN = "income", [
    "gender"], 40, -1, 1, 0, "../Datasets/adult_dataset/processed_adult.csv", 1, 0.001, \
                                     1000, 0, 1, 0, 0

for opt, arg in opts:
    if opt == '-h':
        print(
            "--label_column=<label_column> \n "
            "--protect_columns=<protect_columns> (Only one allowed here) \n"
            "gender - male vs female (protected) \n"
            "race_White - white vs non-white (protected) \n"
            "\n--num_epoch=<num_epoch> \n--id=<id> \n--num_trials=<num_trials> \n"
            "--num_proxies= <num_proxies> \n--file_path=<file_path> \n--verbose=<verbose> \n--lr=<lr> \n "
            "--batch_size=<batch_size> \n"
            "--balance=<balance>\n"
            "Same number of samples in each class if arg is set to 1"
            "--keep=<keep> \n"
            "The proportion of the keep when filtering the majority and minority sets (must be ]0,1])"
            "--filter_maj<filter_maj> --filter_min<filter_min>"
            "1: filters the group to improve model predictions"
            "0: does not filter the group"
            "-1: filters the group to worsen model predictions")
        sys.exit()
    if opt == '--label_column':
        LABEL_COL = int(arg)
    if opt == '--protect_columns':
        PROTECT_COLS = str(arg).split(",")
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
    if opt == '--batch_size':
        BATCH_SIZE = int(arg)
    if opt == '--balance':
        BALANCE = int(arg)
    if opt == '--keep':
        KEEP = float(arg)
    if opt == '--filter_maj':
        FILTER_MAJ = int(arg)
    if opt == '--filter_min':
        FILTER_MIN = int(arg)

if len(PROTECT_COLS) >= 2 or not (0 < KEEP <= 1) or FILTER_MAJ not in [-1, 0, 1] or FILTER_MIN not in [-1, 0, 1]:
    print("Arguments not valid: see flag -h for more information")
    sys.exit(1)

print(
    f"RUNNING SCRIPT WITH ARGUMENTS : -label_column={LABEL_COL} -protect_columns={PROTECT_COLS}"
    f"-num_epoch={NUM_EPOCH} -id={ID} -num_trials={NUM_TRIALS} -num_proxies={NUM_PROXIES}"
    f"-file_path={FILE_PATH} -verbose={VERBOSE} -lr={LR_RATE}"
    f"-batch_size={BATCH_SIZE} -balance={BALANCE} -keep={KEEP} -filter_maj={FILTER_MAJ} -filter_min={FILTER_MIN}")

train_test_datasets = load_split_dataset(FILE_PATH, LABEL_COL,
                                         PROTECT_COLS[0],
                                         is_scaled=True,
                                         num_proxy_to_remove=NUM_PROXIES,
                                         balanced=BALANCE,
                                         keep=KEEP,
                                         verbose=VERBOSE,
                                         filters = [FILTER_MAJ, FILTER_MIN])

tr_maj_d, tr_min_d, tr_d, te_maj_d, te_min_d, te_d = train_test_datasets

print("---------- MAPPING ----------")
print("Train: ", tr_d.mapping)
print("Test: ", te_d.mapping)
print("-----------------------------")

device = torch.device("cpu")
num_predictor_features = tr_maj_d[0][0].shape[0]

# Training data loaders
train_maj_loader = torch.utils.data.DataLoader(dataset=tr_maj_d, batch_size=BATCH_SIZE, shuffle=True)
train_min_loader = torch.utils.data.DataLoader(dataset=tr_min_d, batch_size=BATCH_SIZE, shuffle=True)
train_loader = torch.utils.data.DataLoader(dataset=tr_d, batch_size=BATCH_SIZE, shuffle=True)

test_maj_loader = torch.utils.data.DataLoader(dataset=te_maj_d, batch_size=BATCH_SIZE, shuffle=False)
test_min_loader = torch.utils.data.DataLoader(dataset=te_min_d, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=te_d, batch_size=BATCH_SIZE, shuffle=False)

test_maj_accuracies, test_min_accuracies, test_accuracies, fairness_diffs, base_rate_diff = [], [], [], [], []
for trial in range(NUM_TRIALS):
    print(f"Trial {trial}: Training...")
    # Models
    predictor_maj = Predictor(num_predictor_features).to(device)
    predictor_min = Predictor(num_predictor_features).to(device)
    predictor = Predictor(num_predictor_features).to(device)

    optimizer_maj = optim.Adam(predictor_maj.parameters(), lr=LR_RATE)
    optimizer_min = optim.Adam(predictor_min.parameters(), lr=LR_RATE)
    optimizer = optim.Adam(predictor.parameters(), lr=LR_RATE)

    train_maj_history = train(predictor_maj, device, train_maj_loader, optimizer_maj, NUM_EPOCH, verbose=VERBOSE)
    train_min_history = train(predictor_min, device, train_min_loader, optimizer_min, NUM_EPOCH, verbose=VERBOSE)
    train_history = train_sample_reweight(predictor, device, train_loader, optimizer, NUM_EPOCH, verbose=VERBOSE)

    ###### Test set
    test_maj_pred_labels, test_maj_loss, test_maj_accuracy, _ = test(predictor_maj, device, test_maj_loader)
    test_min_pred_labels, test_min_loss, test_min_accuracy, _ = test(predictor_min, device, test_min_loader)
    test_pred_labels, test_loss, test_accuracy, _ = test(predictor, device, test_loader)

    accs_maj = equalizing_odds(test_maj_pred_labels, test_maj_loader.dataset.label, test_maj_loader.dataset.protect)
    accs_min = equalizing_odds(test_min_pred_labels, test_min_loader.dataset.label, test_min_loader.dataset.protect)
    accs = equalizing_odds(test_pred_labels, test_loader.dataset.label, test_loader.dataset.protect)

    test_maj_accuracies.append(accs_maj)
    test_min_accuracies.append(accs_min)
    test_accuracies.append(accs)

    base_rate_diff.append([[p1 - p2 for p1, p2 in zip(l1, l2)] for l1, l2 in zip(accs_maj, accs_min)])
    fairness_diffs.append([max(acc) - min(acc) for acc in accs])

    #### Saving checkpoint
    if ID >= 0:
        PATH = f"base_rate_models/checkpoints/model_ep_{NUM_EPOCH}/Run_{ID}/trial_{trial}"
        LOSS = "BCELoss"

        os.makedirs(PATH + "/maj/", exist_ok=True)
        torch.save({
            'epoch': NUM_EPOCH,
            'model_state_dict': predictor_maj.state_dict(),
            'optimizer_state_dict': optimizer_maj.state_dict(),
            'loss': LOSS,
        }, PATH + "/maj/checkpoint.pt ")
        train_maj_history.to_pickle(PATH + "/maj/train_history.pkl")

        os.makedirs(PATH + "/min/", exist_ok=True)
        torch.save({
            'epoch': NUM_EPOCH,
            'model_state_dict': predictor_min.state_dict(),
            'optimizer_state_dict': optimizer_min.state_dict(),
            'loss': LOSS,
        }, PATH + "/min/checkpoint.pt ")
        train_maj_history.to_pickle(PATH + "/min/train_history.pkl")

        os.makedirs(PATH + "/combined/", exist_ok=True)
        torch.save({
            'epoch': NUM_EPOCH,
            'model_state_dict': predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
        }, PATH + "/combined/checkpoint.pt ")
        train_history.to_pickle(PATH + "/combined/train_history.pkl")

test_maj_accuracies = np.array(test_maj_accuracies)
test_min_accuracies = np.array(test_min_accuracies)
test_accuracies = np.array(test_accuracies)
fairness_diffs = np.array(fairness_diffs)
base_rate_diff = np.array(base_rate_diff)

print(f"Majority test accuracy: {np.mean(test_maj_accuracies, axis=0)} += {np.std(test_maj_accuracies, axis=0)} \n")
print(f"Minority test accuracy: {np.mean(test_min_accuracies, axis=0)} += {np.std(test_min_accuracies, axis=0)} \n")
print(f"Fairness accuracy: \n {np.mean(test_accuracies, axis=0)} += {np.std(test_accuracies, axis=0)}")
print(f"Base rate difference: \n {np.mean(base_rate_diff, axis=0)} += {np.std(base_rate_diff, axis=0)}")
print(f"Fairness difference: \n {fairness_diffs}")

if ID >= 0:
    PATH = f"base_rate_models/checkpoints/model_ep_{NUM_EPOCH}/Run_{ID}/stats.txt"
    file = open(PATH, "w")
    file.write(
        f"Majority test accuracy:\n {np.mean(test_maj_accuracies, axis=0)} \n+= \n {np.std(test_maj_accuracies, axis=0)} \n")
    file.write(
        f"Minority test accuracy:\n {np.mean(test_min_accuracies, axis=0)} \n+= \n {np.std(test_min_accuracies, axis=0)} \n")
    file.write(f"Fairness accuracy: \n {np.mean(test_accuracies, axis=0)} \n += \n {np.std(test_accuracies, axis=0)}\n")
    file.write(f"Mapping: {te_d.mapping} \n")
    file.write(f"Base rate difference: \n {np.mean(base_rate_diff, axis=0)} \n+=\n {np.std(base_rate_diff, axis=0)}\n")
    file.write(f"Fairness difference: \n {fairness_diffs}")
    file.close()
