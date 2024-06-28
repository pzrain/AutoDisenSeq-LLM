import os
import numpy as np
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import FinetuneDataset
from trainers import FineTrainer
from utils import get_args, EarlyStopping, get_user_seqs_txt, check_path, set_seed, get_item2attribute
from utils import get_args, get_user_seqs_long_txt, get_item2attribute, check_path, set_seed, get_edge_node_op, sample_valid_archs
from ops import PRIMITIVES


def finetune(args, do_eval=False, arch=None):
    print(arch, do_eval)
    set_seed(args.seed)
    output_path = os.path.join(args.output_dir, args.loss_type)
    check_path(output_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = os.path.join(args.data_dir, args.data_name + '.txt')
    user_seq, max_item, valid_rating_matrix, test_rating_matrix, item_frequency = \
        get_user_seqs_txt(args.data_file)
    args.item_size = max_item + 2
    args.mask_id = max_item + 1

    filename = f'{args.model_name}-{args.loss_type}-{args.data_name}-pt_{args.ckp}'
    # save model args
    args.log_file = os.path.join(output_path, filename + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    args.checkpoint_path = os.path.join(output_path, filename + '.pt')

    train_dataset = FinetuneDataset(args, user_seq, item_frequency, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = FinetuneDataset(args, user_seq, item_frequency, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = FinetuneDataset(args, user_seq, item_frequency, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    edgeop, nodeop = get_edge_node_op(PRIMITIVES, args.auto_space)
    from models import DSSRecModel2, DSSRecModel
    # model = DSSRecModel2(args)
    if args.loss_type == 'DSS':
        model = DSSRecModel(args, args.auto_dim, args.auto_head, args.auto_layer, edgeops=edgeop, nodeops=nodeop)
    else:
        model = DSSRecModel2(args, args.auto_dim, args.auto_head, args.auto_layer, edgeops=edgeop, nodeops=nodeop)

    trainer = FineTrainer(model, train_dataloader, eval_dataloader,
                          test_dataloader, args)

    if args.do_eval or do_eval is True:
        print("do_eval")
        
        pretrained_path = os.path.join(output_path, f'pretrain-{args.loss_type}-'
                                                    f'{args.data_name}-epochs-{args.ckp}.pt')
        trainer.load(pretrained_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        if do_eval is True and arch is not None:
            scores, result_info = trainer.valid(0, full_sort=False, arch=arch)
        else:
            scores, result_info = trainer.test(0, full_sort=False, arch=arch)

    else:
        pretrained_path = os.path.join(output_path, f'pretrain-{args.loss_type}-'
                                                    f'{args.data_name}-epochs-{args.ckp}.pt')
        try:
            trainer.load(pretrained_path)
            print(f'Load Checkpoint From {pretrained_path}!')

        except FileNotFoundError:
            print(f'{pretrained_path} Not Found! The Model is same as SASRec')

        early_stopping = EarlyStopping(args.checkpoint_path, patience=1000, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch, full_sort=False)
            early_stopping(np.array([scores[-3]]), trainer.model, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        trainer.args.train_matrix = test_rating_matrix
        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=False)

    print(filename)
    print(result_info)
    print(scores[4])
    with open(args.log_file, 'a') as f:
        f.write(filename + '\n')
        f.write(result_info + '\n')
    return scores[4]


if __name__ == '__main__':
    args = get_args()
    arch = list(eval(args.archs))
    finetune(args, arch=arch)
