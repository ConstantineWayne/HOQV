#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from .bert import BertClf
from .image import ImageClf
from .my_model import HOQV,ce_loss,get_projection_distribution,get_prediction_projection,unified_UCE_loss,GDD_fusion_proj,get_GDD_distribution
MODELS = {
    "bert": BertClf,
    "img": ImageClf,
    'HOQV':HOQV
}


def get_model(args):
    # print(args.model)
    return MODELS[args.model](args)
