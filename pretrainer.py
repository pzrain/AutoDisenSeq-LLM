import torch
from torch.utils.data import DataLoader, RandomSampler

import os

from datasets import PretrainDataset
from utils import get_args, get_user_seqs_long_txt, get_item2attribute, check_path, set_seed, get_edge_node_op, sample_valid_archs
from ops import PRIMITIVES

def pretrain(args):
    set_seed(args.seed)
    output_path = os.path.join(args.output_dir, args.loss_type)
    check_path(output_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # get data
    args.data_file = os.path.join(args.data_dir, args.data_name + '.txt')
    user_seq, max_item, long_sequence = get_user_seqs_long_txt(args.data_file)
    
    # checkpoint
    ckp_file = f'{args.model_name}-{args.loss_type}-{args.data_name}-epochs-{args.ckp}.pt'
    args.checkpoint_path = os.path.join(output_path, ckp_file)

    # number of items and mask
    args.item_size = max_item + 2
    args.mask_id = max_item + 1

    # log the arguments
    log_filename = f'{args.model_name}-{args.loss_type}-{args.data_name}.txt'
    args.log_file = os.path.join(output_path, log_filename)
    print(args)
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    from models import DSSRecModel2, DSSRecModel
    from trainers import DSSPretrainer

    edgeop, nodeop = get_edge_node_op(PRIMITIVES, args.auto_space)
    if args.loss_type == 'DSS':
        model = DSSRecModel(args, args.auto_dim, args.auto_head, args.auto_layer, edgeops=edgeop, nodeops=nodeop)
    else:
        model = DSSRecModel2(args, args.auto_dim, args.auto_head, args.auto_layer, edgeops=edgeop, nodeops=nodeop)
    trainer = DSSPretrainer(model, None, None, None, args)

    # to resume training from last pre-trained epoch
    if os.path.exists(args.checkpoint_path):
        trainer.load(args.checkpoint_path)
        print(f'Resume training from epoch={args.ckp} for pre-training!')
        init_epoch = int(args.ckp) - 1
    else:
        init_epoch = -1
    for epoch in range(args.pre_epochs):
        if epoch <= init_epoch:
            continue

        pretrain_dataset = PretrainDataset(args, user_seq, long_sequence)
        pretrain_sampler = RandomSampler(pretrain_dataset)
        pretrain_dataloader = DataLoader(pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size)

        trainer.pretrain(epoch, pretrain_dataloader, edgeop, nodeop)

        # save checkpoint
        if (epoch + 1) % 20 == 0:
            ckp = f'{args.model_name}-{args.loss_type}-{args.data_name}-epochs-{epoch+1}.pt'
            checkpoint_path = os.path.join(output_path, ckp)
            trainer.save(checkpoint_path)


if __name__ == '__main__':
    args = get_args()
    pretrain(args)
