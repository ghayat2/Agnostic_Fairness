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
                                "batch_size=", "balance="])
except getopt.GetoptError:
    print("Wrong format, see -h for help ...")
    print(sys.argv)
    sys.exit(2)

# INPUT PARAMS
LABEL_COL, PROTECT_COLS, NUM_EPOCH, ID, NUM_TRIALS, NUM_PROXIES, FILE_PATH, VERBOSE, \
LR_RATE, BATCH_SIZE, BALANCE = "income", ["gender"], 40, -1, 1, 0, "../Datasets/adult_dataset/processed_adult.csv", 1, 0.001, \
                      1000, 0

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
            "Same number of samples in each class if arg is set to 1")
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

if len(PROTECT_COLS) >= 2:
    print("Arguments not valid: see flag -h for more information")
    sys.exit(1)

print(
    f"RUNNING SCRIPT WITH ARGUMENTS : -label_column={LABEL_COL} -protect_columns={PROTECT_COLS}"
    f"-num_epoch={NUM_EPOCH} -id={ID} -num_trials={NUM_TRIALS} -num_proxies={NUM_PROXIES}"
    f"-file_path={FILE_PATH} -verbose={VERBOSE} -lr={LR_RATE}"
    f"-batch_size={BATCH_SIZE} -balance={BALANCE}")

train_maj_dataset, test_maj_dataset, train_min_dataset, test_min_dataset = load_split_dataset(FILE_PATH, LABEL_COL,
                                                                                              PROTECT_COLS[0],
                                                                                              is_scaled=True,
                                                                                              num_proxy_to_remove=NUM_PROXIES,
                                                                                              balanced=BALANCE)

device = torch.device("cpu")
num_predictor_features = train_maj_dataset[0][0].shape[0]

train_maj_dataset = filter(train_maj_dataset, num_predictor_features, improve=False, epochs=NUM_EPOCH, keep=0.8)
train_min_dataset = filter(train_min_dataset, num_predictor_features, improve=True, epochs=NUM_EPOCH, keep=0.9)

# Data loaders
train_maj_loader = torch.utils.data.DataLoader(dataset=train_maj_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_maj_loader = torch.utils.data.DataLoader(dataset=test_maj_dataset, batch_size=BATCH_SIZE, shuffle=False)
train_min_loader = torch.utils.data.DataLoader(dataset=train_min_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_min_loader = torch.utils.data.DataLoader(dataset=test_min_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_maj_accuracies, test_min_accuracies, base_rate_diff = [], [], []
for trial in range(NUM_TRIALS):
    print(f"Trial {trial}: Training...")
    # Model and Optimizer
    predictor_maj = Predictor(num_predictor_features).to(device)
    optimizer_maj = optim.Adam(predictor_maj.parameters(), lr=LR_RATE)

    predictor_min = Predictor(num_predictor_features).to(device)
    optimizer_min = optim.Adam(predictor_min.parameters(), lr=LR_RATE)

    train_maj_history = train(predictor_maj, device, train_maj_loader, optimizer_maj, NUM_EPOCH, verbose=VERBOSE)
    train_min_history = train(predictor_min, device, train_min_loader, optimizer_min, NUM_EPOCH, verbose=VERBOSE)

    ###### Test set
    test_maj_pred_labels, test_maj_loss, test_maj_accuracy, _ = test(predictor_maj, device, test_maj_loader)
    test_min_pred_labels, test_min_loss, test_min_accuracy, _ = test(predictor_min, device, test_min_loader)

    accs_maj = equalizing_odds(test_maj_pred_labels, test_maj_loader.dataset.label, test_maj_loader.dataset.protect)
    accs_min = equalizing_odds(test_min_pred_labels, test_min_loader.dataset.label, test_min_loader.dataset.protect)
    test_maj_accuracies.append(accs_maj)
    test_min_accuracies.append(accs_min)
    base_rate_diff.append([[p1 - p2 for p1, p2 in zip(l1, l2)] for l1, l2 in zip(accs_maj, accs_min)])

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

test_maj_accuracies = np.array(test_maj_accuracies)
test_min_accuracies = np.array(test_min_accuracies)
base_rate_diff = np.array(base_rate_diff)

print(f"Majority test accuracy: {np.mean(test_maj_accuracies, axis=0)} += {np.std(test_maj_accuracies, axis=0)} \n")
print(f"Minority test accuracy: {np.mean(test_min_accuracies, axis=0)} += {np.std(test_min_accuracies, axis=0)} \n")
print(f"Base rate difference: \n {np.mean(base_rate_diff, axis=0)} += {np.std(base_rate_diff, axis=0)}")

if ID >= 0:
    PATH = f"base_rate_models/checkpoints/model_ep_{NUM_EPOCH}/Run_{ID}/stats.txt"
    file = open(PATH, "w")
    file.write(f"Majority test accuracy: {np.mean(test_maj_accuracies, axis=0)} += {np.std(test_maj_accuracies, axis=0)} \n")
    file.write(f"Minority test accuracy: {np.mean(test_min_accuracies, axis=0)} += {np.std(test_min_accuracies, axis=0)} \n")
    file.write(f"Base rate difference: \n {np.mean(base_rate_diff, axis=0)} += {np.std(base_rate_diff, axis=0)}")
    file.close()
