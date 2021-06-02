#!/usr/bin/env python
# coding: utf-8

"""Resnet Random weight allocation

This script implements case 3 of the document: https://mit.zoom.us/j/98140955616
It clusters the training (Bias_0.8) training data into 5 groups, one of them containnig the protceted group
(either female for doctors or male for nurses). During model training, we assign the same weight to every group except
for one, where the weight is 4 times the others. We hope that this procedure will increase the fairness performance from
 Case 1 (Normal training on Bias_0.8 dataset)"""

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
                                "visdom=", "id=", "num_trials=", "dataset="])
except getopt.GetoptError:
    print("Wrong format ...")
    print(sys.argv)
    sys.exit(2)

W_PROTECTED, BIAS, VAL_MODE, START_EPOCH, NUM_EPOCH, NUM_CLUSTERS, SHOW_PROGRESS, ID, DATASET = 1, 0, False, 0, 15, 5, False, 0, "doctor_nurse"

for opt, arg in opts:
    if opt == '-h':
        print("Case_3.py --w_protected=<w_protected> --bias=<bias> --val_mode=<val_mode> --start_epoch=<start_epoch>"
              "--num_epoch=<num_epoch> --num_clusters=<num_clusters> --visdom=<visdom> --id=<id> "
              "--num_trials=<num_trials> --dataset=<dataset>")
        sys.exit()
    if opt == '--w_protected':
        W_PROTECTED = int(arg)
    if opt == '--bias':
        BIAS = float(arg)
    if opt == '--val_mode':
        VAL_MODE = int(arg)
    if opt == '--start_epoch':
        START_EPOCH = int(arg)
    if opt == '--num_epoch':
        NUM_EPOCH = int(arg)
    if opt == '--num_clusters':
        NUM_CLUSTERS = int(arg)
    if opt == '--visdom':
        SHOW_PROGRESS = int(arg)
    if opt == '--id':
        ID = int(arg)
    if opt == '--num_trials':
        NUM_TRIALS = int(arg)
    if opt == '--dataset':
        DATASET = str(arg)

if DATASET not in ["doctor_nurse", "basket_volley"]:
    print("Invalid arguments, exiting ...")
    sys.exit(1)

print(
    f"RUNNING SCRIPT WITH ARGUMENTS : -w_protected={W_PROTECTED} -bias={BIAS} -val_mode={VAL_MODE} -start_epoch={START_EPOCH} "
    f"-num_epoch={NUM_EPOCH}, -num_clusters={NUM_CLUSTERS}, -visdom={SHOW_PROGRESS}, -id={ID}, -num_trials={NUM_TRIALS} -dataset={DATASET}")

data_dir = '../Datasets/doctor_nurse/train_test_split' if DATASET == "doctor_nurse" else \
    '../Datasets/basket_volley/train_test_split'

class0_min, class1_min = dr_f_d + dr_f_l if DATASET == "doctor_nurse" else bask_y_m + bask_y_f, \
                         nur_m_d + nur_m_l if DATASET == "doctor_nurse" else voll_r_m + voll_r_f
class0, class1 = "basket", "volley"
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

image_datasets = {x: my_ImageFolderRandomCluster(os.path.join(data_dir, f"train_{BIAS}" if x == "train" and BIAS else x),
                                                 data_transforms[x],
                                                 make_clusters([os.listdir(os.path.join(data_dir,
                                                                                  f"train_{BIAS}/" + class0 if x == "train" and BIAS else x)),
                                                          os.listdir(os.path.join(data_dir,
                                                                                  f"train_{BIAS}/" + class1 if x == "train" and BIAS else x))],
                                                         [set(dr_f_d + dr_f_l), set(nur_m_l + nur_m_d)]
                                                         if DATASET == "doctor_nurse" else
                                                         [set(nur_m_d + nur_m_l), set(voll_r_m + voll_r_f)],
                                                         K=NUM_CLUSTERS))
                                                                for x in ['train', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# For descirption of architecture of resNet, see: https://arxiv.org/pdf/1512.03385.pdf

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
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    if START_EPOCH:
        PATH = "Case_3/checkpoints/" + (
            "w_val" if VAL_MODE else "w.o_val") + f"/Bias_{BIAS}/model_ep_{START_EPOCH}/Run_{ID}/trial_{trial}/checkpoint.pt"
        checkpoint = torch.load(PATH)
        model_conv.load_state_dict(checkpoint['model_state_dict'])
        optimizer_conv.load_state_dict(checkpoint['optimizer_state_dict'])
        exp_lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    model_conv = train_model_groups(model_conv, criterion, optimizer_conv, exp_lr_scheduler, W_PROTECTED, dataloaders,
                                    dataset_sizes, device,
                                    start_epoch=START_EPOCH,
                                    num_epochs=NUM_EPOCH, num_clusters=NUM_CLUSTERS,
                                    val_mode=VAL_MODE, show_progress=SHOW_PROGRESS)

    train_accs.append(round(float(accuracy(model_conv, device, dataloaders['train'])), 3))
    test_accs.append(round(float(accuracy(model_conv, device, dataloaders['test'])), 3))
    fairness_accs.append(
        demographic_parity(model_conv, device, image_datasets["test"], [class0_min, class1_min]).to_numpy())

    # #### Saving checkpoint

    PATH = "Case_3/checkpoints/" + (
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

train_accs = np.array(train_accs)
test_accs = np.array(test_accs)
fairness_accs = np.array(fairness_accs)
PATH = "Case_3/checkpoints/" + (
    "w_val" if VAL_MODE else "w.o_val") + f"/Bias_{BIAS}/model_ep_{START_EPOCH + NUM_EPOCH}/Run_{ID}/stats.txt"

file = open(PATH, "w")
file.write(f"Training accuracy: {train_accs.mean()} += {train_accs.std()} \n")
file.write(f"Test accuracy: {test_accs.mean()} += {test_accs.std()} \n")
file.write(f"Fairness accuracy: \n {np.mean(fairness_accs, axis=0)} \n += \n {np.std(fairness_accs, axis=0)}")
file.close()
