# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import NamedTuple, Tuple

import torch
from torch import Tensor
import random
import math


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class RandomMaskingOutput(NamedTuple):
    x_masked: Tensor
    mask: Tensor
    ids_restore: Tensor
    ids_keep: Tensor


def random_masking(
    x: torch.Tensor,
    mask_ratio: float,
) -> RandomMaskingOutput:
    """
    Original paper: https://arxiv.org/pdf/2111.06377.pdf
    OSS implementation: https://github.com/facebookresearch/mae/blob/main/models_mae.py#L123
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    n, l, d = x.shape  # batch, length, dim
    len_keep = int(l * (1 - mask_ratio))

    noise = torch.rand(n, l, device=x.device)  # noise in [0, 1]

    assert len_keep >= 1, "must keep at least 1 patch"

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([n, l], device=x.device)
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return RandomMaskingOutput(
        x_masked=x_masked,
        mask=mask,
        ids_restore=ids_restore,
        ids_keep=ids_keep,
    )


def random_masking_2d(
    x: torch.Tensor,
    mask_ratio_h: float,
    mask_ratio_w: float,
    num_patches_h: int,
    num_patches_w: int,
) -> Tensor:
    """
    Perform 2d masking as described in audio mae paper https://arxiv.org/pdf/2207.06405.pdf
    Code adapted from https://github.com/facebookresearch/AudioMAE/blob/main/models_vit.py#L88
    Args:
        x: Input tensor containing patches of shape bsz x seq_len x embed_dim
        mask_ratio_h: masking ratio for height dimension
        mask_ratio_w: masking ratio for width dimension
        num_patches_h: number of patches in height dimension
        num_patches_w: number of patches in width dimension
    """
    n, _, d = x.shape

    x = x.reshape(n, num_patches_h, num_patches_w, d)
    x_masked, len_keep_h = _random_masking_1d(
        x, mask_ratio_h, num_patches_h, num_patches_w
    )
    x_masked = x_masked.transpose(1, 2)
    x_masked, len_keep_w = _random_masking_1d(
        x_masked, mask_ratio_w, num_patches_w, len_keep_h
    )
    x_masked = x_masked.transpose(1, 2)
    x_masked = x_masked.reshape(n, len_keep_h * len_keep_w, d)

    return x_masked


def _random_masking_1d(
    x: Tensor,
    mask_ratio: float,
    num_patches_h: int,
    num_patches_w: int,
) -> Tuple[Tensor, int]:
    # x shape : bsz x h x w x embed_dim
    n, _, _, d = x.shape
    len_keep = int(num_patches_h * (1 - mask_ratio))
    noise = torch.rand(n, num_patches_h, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, num_patches_w, d)
    x_masked = torch.gather(x, dim=1, index=index)
    return x_masked, len_keep
