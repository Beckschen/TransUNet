import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from common_parser import get_common_parser, dataset_config


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(
        base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("%s test iterations per epoch", len(testloader))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = (
            sampled_batch["image"],
            sampled_batch["label"],
            sampled_batch["case_name"][0],
        )
        metric_i = test_single_volume(
            image,
            label,
            model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=args.z_spacing,
        )
        metric_list += np.array(metric_i)
        logging.info(
            "idx %s case %s mean_dice %s mean_hd95 %s",
            i_batch,
            case_name,
            np.mean(metric_i, axis=0)[0],
            np.mean(metric_i, axis=0)[1],
        )
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info(
            "Mean class %d mean_dice %f mean_hd95 %f",
            i,
            metric_list[i - 1][0],
            metric_list[i - 1][1],
        )
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info(
        "Testing performance in best val model: mean_dice : %f mean_hd95 : %f",
        performance,
        mean_hd95,
    )
    return "Testing Finished!"


def main():
    parser = get_common_parser(state="test")
    args = parser.parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.volume_path = dataset_config[dataset_name]["volume_path"]
    args.Dataset = dataset_config[dataset_name]["Dataset"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]
    args.z_spacing = dataset_config[dataset_name]["z_spacing"]
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = "TU_" + dataset_name + str(args.img_size)
    snapshot_path = "/project/mhssain9/model/{}/{}".format(args.exp, "TU")
    snapshot_path = snapshot_path + "_pretrain" if args.is_pretrain else snapshot_path
    snapshot_path += "_" + args.vit_name
    snapshot_path = snapshot_path + "_skip" + str(args.n_skip)
    snapshot_path = (
        snapshot_path + "_vitpatch" + str(args.vit_patches_size)
        if args.vit_patches_size != 16
        else snapshot_path
    )
    snapshot_path = (
        snapshot_path + "_epo" + str(args.max_epochs)
        if args.max_epochs != 30
        else snapshot_path
    )
    if (
        dataset_name == "ACDC"
    ):  # using max_epoch instead of iteration to control training duration
        snapshot_path = (
            snapshot_path + "_" + str(args.max_iterations)[0:2] + "k"
            if args.max_iterations != 30000
            else snapshot_path
        )
    snapshot_path = snapshot_path + "_bs" + str(args.batch_size)
    snapshot_path = (
        snapshot_path + "_lr" + str(args.base_lr)
        if args.base_lr != 0.01
        else snapshot_path
    )
    snapshot_path = snapshot_path + "_" + str(args.img_size)
    snapshot_path = (
        snapshot_path + "_s" + str(args.seed) if args.seed != 1234 else snapshot_path
    )

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find("R50") != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size),
        )
    net = ViT_seg(
        config_vit, img_size=args.img_size, num_classes=config_vit.n_classes
    ).cuda()

    snapshot = os.path.join(snapshot_path, "best_model.pth")
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace("best_model", "epoch_" + str(args.max_epochs - 1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split("/")[-1]

    log_folder = "./test_log/test_log_" + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=log_folder + "/" + snapshot_name + ".txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = "/project/mhssain9/predictions"
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


if __name__ == "__main__":
    main()
