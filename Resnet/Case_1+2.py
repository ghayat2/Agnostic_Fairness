#!/usr/bin/env python
# coding: utf-8

"""ResNet Vanilla feature extractor

This script aims to train a classifier based on representations extracted from a pretrained resNet. The goal is to assess
the performance of the classifier when no reweighting is done on the dataset.
"""

from __future__ import print_function, division

from torch.utils.data import TensorDataset

from fairness_metrics import *
from model import *
from my_ImageFolder import *

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

try:
    opts, args = getopt.getopt(sys.argv[1:], "h",
                               ["w_protected=", "bias=", "val_mode=", "start_epoch=", "num_epoch=", "visdom=", "id=",
                                "num_trials=", "dataset="])
except getopt.GetoptError:
    print("Wrong format ...")
    print(sys.argv)
    sys.exit(2)

W_PROTECTED, BIAS, VAL_MODE, START_EPOCH, NUM_EPOCH, SHOW_PROGRESS, ID, DATASET, NUM_TRIALS = 1, 0.8, False, 0, 15, False, 0, "doctor_nurse", 1

for opt, arg in opts:
    if opt == '-h':
        print("Case_1+2.py --w_protected=<w_protected> --bias=<bias> --val_mode=<val_mode> --start_epoch=<start_epoch>"
              "--num_epoch=<num_epoch> --visdom=<wisdom> --id=<id> --num_trials=<num_trials> --dataset=<dataset>")
        sys.exit(0)
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
    f"-num_epoch={NUM_EPOCH} -visdom={SHOW_PROGRESS}, -id={ID}, -num_trials={NUM_TRIALS} -dataset={DATASET}")

data_dir = '../Datasets/doctor_nurse/train_test_split' if DATASET == "doctor_nurse" else \
    '../Datasets/basket_volley/train_test_split'

class0_min, class1_min = dr_f_d + dr_f_l if DATASET == "doctor_nurse" else bask_y_m + bask_y_f, \
                         nur_m_d + nur_m_l if DATASET == "doctor_nurse" else voll_r_m + voll_r_f

class0_maj, class1_maj = dr_m_d + dr_m_l if DATASET == "doctor_nurse" else bask_r_m + bask_r_f, \
                         nur_f_d + nur_f_l if DATASET == "doctor_nurse" else voll_y_m + voll_y_f
protected_groups = set(class0_min + class1_min)

###  Defining dataloaders

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

image_datasets = {
    x: my_ImageFolder(os.path.join(data_dir, f"train_{BIAS}" if x == "train" and BIAS else x),
                             data_transforms[x],
                             protected_groups, W_PROTECTED)
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
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    if START_EPOCH:
        PATH = ("Case_2/" if W_PROTECTED != 1 else "Case_1/") + "checkpoints/" + (
            "w_val" if VAL_MODE else "w.o_val") + f"/Bias_{BIAS}/model_ep_{START_EPOCH}/Run_{ID}/trial_{trial}/checkpoint.pt "
        checkpoint = torch.load(PATH)
        model_conv.load_state_dict(checkpoint['model_state_dict'])
        optimizer_conv.load_state_dict(checkpoint['optimizer_state_dict'])
        exp_lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, dataset_sizes,
                             device,
                             start_epoch=START_EPOCH,
                             num_epochs=NUM_EPOCH,
                             val_mode=VAL_MODE, show_progress=SHOW_PROGRESS)

    print("Old way: ")
    train_accs.append(float(accuracy(model_conv, device, dataloaders['train'])))
    test_accs.append(float(accuracy(model_conv, device, dataloaders['test'])))

    fairness_accs.append(
        demographic_parity(model_conv, device, image_datasets["test"], [class0_min, class1_min]).to_numpy())

    # #### Saving checkpoint

    PATH = ("Case_2/" if W_PROTECTED != 1 else "Case_1/") + "checkpoints/" + (
        "w_val" if VAL_MODE else "w.o_val") + f"/Bias_{BIAS}/model_ep_{START_EPOCH + NUM_EPOCH}/Run_{ID}/trial_{trial}"
    LOSS = "CrossEntropyLoss"

    if ID >= 0:
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
PATH = ("Case_2/" if W_PROTECTED != 1 else "Case_1/") + "checkpoints/" + (
    "w_val" if VAL_MODE else "w.o_val") + f"/Bias_{BIAS}/model_ep_{START_EPOCH + NUM_EPOCH}/Run_{ID}/stats.txt"

if ID >= 0:
    file = open(PATH, "w")
    file.write(f"Training accuracy: {train_accs.mean()} += {train_accs.std()} \n")
    file.write(f"Test accuracy: {test_accs.mean()} += {test_accs.std()} \n")
    file.write(f"Fairness accuracy: \n {np.mean(fairness_accs, axis=0)} \n += \n {np.std(fairness_accs, axis=0)}")
    file.close()
