import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size

from .bases import ImageDataset, TextDataset, ImageTextMLMDataset

from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid
from .cuhkpedesM import CUHKPEDESM

__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid, "CUHK-PEDES-M":CUHKPEDESM}
def build_transforms4grayscale(img_size=(384, 128), mean=[0.48145466, 0.4578275, 0.40821073], std = [0.26862954, 0.26130258, 0.27577711]):
    mean = [float(x) for x in mean]
    std = [float(x) for x in mean]
    height, width = img_size
    return T.Compose([
            T.Grayscale(num_output_channels=3), 
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            # T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
def build_transforms(img_size=(384, 128), aug=False, is_train=True, mean=[0.48145466, 0.4578275, 0.40821073], std = [0.26862954, 0.26130258, 0.27577711]):
    height, width = img_size
    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform
    mean = [float(x) for x in mean]
    std = [float(x) for x in mean]
    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            # T.Pad(10),
            # T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
             batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict

def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("DANK.dataset")

    num_workers = args.trainer.num_workers
    dataset = __factory[args.dataloader.dataset_name](root=args.iocfg.datadir)
    num_classes = len(dataset.train_id_container)
    logger.info(f"{args.dataloader.dataset_name} has  {num_classes} classes")
    if args.training:
        grayscale_transform = build_transforms4grayscale(img_size=args.image_encoder.img_size,
                                            mean=args.dataloader.transform.mean,
                                            std=args.dataloader.transform.std)
        train_transforms = build_transforms(img_size=args.image_encoder.img_size,
                                            aug=True,
                                            is_train=True, 
                                            mean=args.dataloader.transform.mean,
                                            std=args.dataloader.transform.std)
        mim_train_transforms = build_transforms(img_size=args.image_encoder.img_size,
                                            aug=False,
                                            is_train=True, 
                                            mean=args.dataloader.transform.mean,
                                            std=args.dataloader.transform.std)
        val_transforms = build_transforms(img_size=args.image_encoder.img_size,
                                        is_train=False,
                                        mean=args.dataloader.transform.mean,
                                        std=args.dataloader.transform.std)


        train_set = ImageTextMLMDataset(dataset.train,
                                    train_transforms,
                                    text_length=args.dataloader.text_length,
                                    grayscale_transform=grayscale_transform,
                                    mim_transform=mim_train_transforms,
                                    enhance_mlm = args.losses.mlm.use_custom,
                                    prob4newmlm    = args.losses.mlm.mask_prob)

        if args.dataloader.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.dataloader.batch_size // get_world_size()
                # TODO wait to fix bugs
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.dataloader.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)

            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.dataloader.batch_size}, id: {args.dataloader.batch_size // args.dataloader.num_instance}, instance: {args.dataloader.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.dataloader.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.dataloader.batch_size,
                                              args.dataloader.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate)
        elif args.dataloader.sampler == 'random':
            # TODO add distributed condition
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.dataloader.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # use test set as validate set
        ds = dataset.val if args.dataloader.val_dataset == 'val' else dataset.test
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                   val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'],
                                  ds['captions'],
                                  text_length=args.dataloader.text_length)

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.dataloader.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.dataloader.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:
        # build dataloader for testing
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.image_encoder.img_size,
                                            is_train=False,
                                            mean=args.dataloader.transform.mean,
                                            std=args.dataloader.transform.std)

        ds = dataset.test
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.dataloader.text_length)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.dataloaer.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.dataloader.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        return test_img_loader, test_txt_loader, num_classes
