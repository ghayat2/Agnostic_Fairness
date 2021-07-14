from __future__ import print_function, division
from torch.utils.data import TensorDataset
import pandas as pd
from model import *
from my_ImageFolder import *
from fairness_metrics import *

plt.ion()  # interactive mode

path_dr_f_d = '../Datasets/doctor_nurse/dr/fem_dr_dark_56/'
path_dr_f_l = '../Datasets/doctor_nurse/dr/fem_dr_light_256/'
path_dr_m_d = '../Datasets/doctor_nurse/dr/mal_dr_dark_62/'
path_dr_m_l = '../Datasets/doctor_nurse/dr/mal_dr_light_308/'

dr_f_d = os.listdir(path_dr_f_d)
dr_f_l = os.listdir(path_dr_f_l)
dr_m_d = os.listdir(path_dr_m_d)
dr_m_l = os.listdir(path_dr_m_l)

path_nur_f_d = '../Datasets/doctor_nurse/nurse/fem_nurse_dark_63/'
path_nur_f_l = '../Datasets/doctor_nurse/nurse/fem_nurse_light_252/'
path_nur_m_d = '../Datasets/doctor_nurse/nurse/mal_nurse_dark_76/'
path_nur_m_l = '../Datasets/doctor_nurse/nurse/mal_nurse_light_203/'

nur_f_d = os.listdir(path_nur_f_d)
nur_f_l = os.listdir(path_nur_f_l)
nur_m_d = os.listdir(path_nur_m_d)
nur_m_l = os.listdir(path_nur_m_l)

path_bask_r_f = '../Datasets/basket_volley/basket/basket_f_r/'
path_bask_y_f = '../Datasets/basket_volley/basket/basket_f_y/'
path_bask_r_m = '../Datasets/basket_volley/basket/basket_m_r/'
path_bask_y_m = '../Datasets/basket_volley/basket/basket_m_y/'

bask_r_f = os.listdir(path_bask_r_f)
bask_y_f = os.listdir(path_bask_y_f)
bask_r_m = os.listdir(path_bask_r_m)
bask_y_m = os.listdir(path_bask_y_m)

path_voll_r_f = '../Datasets/basket_volley/volley/volley_f_r/'
path_voll_y_f = '../Datasets/basket_volley/volley/volley_f_y/'
path_voll_r_m = '../Datasets/basket_volley/volley/volley_m_r/'
path_voll_y_m = '../Datasets/basket_volley/volley/volley_m_y/'

voll_r_f = os.listdir(path_voll_r_f)
voll_y_f = os.listdir(path_voll_y_f)
voll_r_m = os.listdir(path_voll_r_m)
voll_y_m = os.listdir(path_voll_y_m)

dr_m, dr_f = len(dr_m_d) + len(dr_m_l), len(dr_f_d) + len(dr_f_l)

try:
    opts, args = getopt.getopt(sys.argv[1:], "h",
                               ["w_protected=", "bias=", "val_mode=", "start_epoch=", "num_epoch=", "num_clusters=",
                                "id=", "num_trials=", "dataset=", "update=", "update_lr=", "clusters="])
except getopt.GetoptError:
    print("Wrong format ...")
    print(sys.argv)
    sys.exit(2)

W_PROTECTED, BIAS, VAL_MODE, START_EPOCH, NUM_EPOCH, NUM_CLUSTERS, ID, DATASET, NUM_TRIALS, UPDATE_LR, UPDATE, CLUSTERS = \
    1, 0, False, 0, 15, 2, 0, "doctor_nurse", 1, 1, "cluster", None

for opt, arg in opts:
    if opt == '-h':
        print(
            "--w_protected=<w_protected> \n"
            "This parameter is only relevant when implementing the preprocessing reweighting method: \n"
            "Images from the minority will be applied a weight of <w_protected> when training \n"
            "--bias=<bias> \n"
            "This attribute specifies the level of bias present in the training set\n"
            "For the moment, bias can take values in {0,0.8}, where bias > 0 specifies the ratio of majority/minority\n"
            "present in the training set\n"
            "--val_mode=<val_mode>\n"
            "Whether to train the model with validation\n"
            "--start_epoch=<start_epoch>\n"
            "Whether to start training a model from scratch or from a certain epoch \n"
            "--update=<update> \n"
            "cluster: each cluster has a weight \n"
            "sample: each sample has a weight \n"
            "individual: each sample is treated as an independent individual\n"
            "--num_epoch=<num_epoch> \n--id=<id> \n--num_trials=<num_trials> \n"
            "--update_lr=<update_lr> \n"
            "--clusters=<clusters>"
            "This parameter is the name of a python dictionary mapping each sample to its cluster")

        sys.exit()
    if opt == '--w_protected':
        W_PROTECTED = int(arg)
    if opt == '--bias':
        BIAS = float(arg)
    if opt == '--val_mode':
        VAL_MODE = int(arg)
    if opt == '--clusters':
        CLUSTERS = str(arg)
    if opt == '--start_epoch':
        START_EPOCH = int(arg)
    if opt == '--num_epoch':
        NUM_EPOCH = int(arg)
    if opt == '--num_clusters':
        NUM_CLUSTERS = int(arg)
    if opt == '--id':
        ID = int(arg)
    if opt == '--num_trials':
        NUM_TRIALS = int(arg)
    if opt == '--dataset':
        DATASET = str(arg)
    if opt == '--update':
        UPDATE = str(arg)
    if opt == '--update_lr':
        UPDATE_LR = float(arg)

if DATASET not in ["doctor_nurse", "basket_volley"] or UPDATE not in ["cluster", "sample", "individual"]:
    print("Invalid arguments, exiting ...")
    sys.exit(1)

print(
    f"RUNNING SCRIPT WITH ARGUMENTS : -w_protected={W_PROTECTED} -bias={BIAS} -val_mode={VAL_MODE} -start_epoch={START_EPOCH} "
    f"-num_epoch={NUM_EPOCH}, -num_clusters={NUM_CLUSTERS}, -id={ID}, -num_trials={NUM_TRIALS} -dataset={DATASET}"
    f"-update={UPDATE}, -update_lr={UPDATE_LR} -clusters={CLUSTERS}")

data_dir = '../Datasets/doctor_nurse/train_test_split' if DATASET == "doctor_nurse" else \
    '../Datasets/basket_volley/train_test_split'

class0_min, class1_min = dr_f_d + dr_f_l if DATASET == "doctor_nurse" else bask_y_m + bask_y_f, \
                         nur_m_d + nur_m_l if DATASET == "doctor_nurse" else voll_r_m + voll_r_f

class0_maj, class1_maj = dr_m_d + dr_m_l if DATASET == "doctor_nurse" else bask_r_m + bask_r_f, \
                         nur_f_d + nur_f_l if DATASET == "doctor_nurse" else voll_y_m + voll_y_f

protected_groups = set(class0_min + class1_min)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: my_ImageFolderCluster(os.path.join(data_dir, f"train_{BIAS}" if x == "train" and BIAS else x),
                                           data_transforms[x],
                                           [class0_maj + class1_min, class1_maj + class0_min],
                                           CLUSTERS if x == "train" else None)
                  for x in ['train', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'test']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_accs, test_accs, fairness_accs = [], [], []
for trial in range(NUM_TRIALS):
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, len(class_names))

    model_conv = model_conv.to(device)

    criterion = weighted_cross_entropy_loss  # nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.2)

    if START_EPOCH:
        PATH = "Case_3/checkpoints/" + (
            "w_val" if VAL_MODE else "w.o_val") + f"/Bias_{BIAS}/model_ep_{START_EPOCH}/Run_{ID}/trial_{trial}/checkpoint.pt"
        checkpoint = torch.load(PATH)
        model_conv.load_state_dict(checkpoint['model_state_dict'])
        optimizer_conv.load_state_dict(checkpoint['optimizer_state_dict'])
        exp_lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    func = train_cluster_reweight if UPDATE == "cluster" else \
        (train_sample_reweight if UPDATE == "sample" else train_individual_reweight)
    history = func(model_conv, device, dataloaders["train"], optimizer_conv, exp_lr_scheduler, NUM_EPOCH,
                   num_clusters=NUM_CLUSTERS, num_labels=2, update_lr=UPDATE_LR)

    ###### Test set
    train_pred_labels, train_labels, train_protect, _, train_accuracy, _ = test(model_conv, device,
                                                                                dataloaders["train"])
    test_pred_labels, test_labels, test_protect, _, test_accuracy, _ = test(model_conv, device, dataloaders["test"])

    train_accs.append(train_accuracy)
    test_accs.append(test_accuracy)

    accs = equalizing_odds(test_pred_labels, test_labels, test_protect)
    fairness_accs.append(accs)

    # #### Saving checkpoint
    if ID >= 0:
        PATH = "Reweighting/checkpoints/" + ("cluster_update/" if UPDATE == "cluster" else
                                         ("sample_update/" if UPDATE == "sample" else "individual_update/")) + (
            "w_val" if VAL_MODE else "w.o_val") + f"/Bias_{BIAS}/model_ep_{START_EPOCH + NUM_EPOCH}/Run_{ID}/trial_{trial}"
        LOSS = "CrossEntropyLoss"

        os.makedirs(PATH, exist_ok=True)
        torch.save({
            'epoch': START_EPOCH + NUM_EPOCH,
            'model_state_dict': model_conv.state_dict(),
            'optimizer_state_dict': optimizer_conv.state_dict(),
            'lr_scheduler_state_dict': exp_lr_scheduler.state_dict(),
            'loss': LOSS,
        }, PATH + "/checkpoint.pt")

if ID >= 0:
    train_accs = np.array(train_accs)
    test_accs = np.array(test_accs)
    fairness_accs = np.array(fairness_accs)
    PATH = "Reweighting/checkpoints/" + ("cluster_update/" if UPDATE == "cluster" else
                                         ("sample_update/" if UPDATE == "sample" else "individual_update/")) + (
        "w_val" if VAL_MODE else "w.o_val") + f"/Bias_{BIAS}/model_ep_{START_EPOCH + NUM_EPOCH}/Run_{ID}/stats.txt"

    file = open(PATH, "w")
    file.write(f"Test accuracy: {test_accs.mean()} += {test_accs.std()} \n")
    file.write(f"Fairness accuracy: \n {np.mean(fairness_accs, axis=0)} \n += \n {np.std(fairness_accs, axis=0)}")
    file.close()
