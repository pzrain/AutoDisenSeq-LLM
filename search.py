'''
search for best models given supernet using evolution algorithms
'''

import os

import numpy as np
from tqdm import tqdm
from queue import Queue
import threading
import time

import torch
from torch.utils.data import DataLoader

from utils import get_edge_node_op, sample_valid_archs, mutate_arch, get_args
from ops import PRIMITIVES
from finetuner import finetune

def run_exps(devices, jobs, block=True, interval=0.1):
    '''
    jobs:
      - func
      - kwargs
      - callback
    '''
    job_queue = jobs
    dev_queue = devices
    
    pool = mp.Pool()

    def gen_callback(dev, calls):
        def callback(res):
            dev_queue.put(dev)
            return calls(res)
        return callback

    def gen_err_callback(dev, func):
        def callback(err):
            dev_queue.put(dev)
            return func(err)
        return callback

    def run_job():
        while True:
            job = job_queue.get()
            if job == False:
                break
            dev = dev_queue.get()
            pool.apply_async(
                job['func'],
                kwds={**job['kwargs'], 'device': dev},
                callback=gen_callback(dev, job['callback']),
                error_callback=gen_err_callback(dev, job['err'] if 'err' in job else lambda x: x)
            )
            time.sleep(interval)
    
    th = threading.Thread(target=run_job)
    th.daemon = True
    th.start()
    if block:
        th.join()
        pool.close()
        pool.join()
    return th, pool

def wrap_queue(lists):
    queue = Queue()
    for ele in lists: queue.put(ele)
    queue.put(False)
    return queue

def eval_model(model, arch, loaders, device):
    model.eval()
    gt = []
    pred = []
    with torch.no_grad():
        for batch in loaders:
            logit = model(batch[0].to(device), batch[2].to(device), arch)
            pred.extend(logit.argmax(1).detach().cpu().tolist())
            gt.extend(batch[1].cpu().tolist())
    return (np.array(gt) == np.array(pred)).mean()


def eval_param_archs(args, archs):
    
    perf = []
    for a in archs:
        score = finetune(args, do_eval=True, arch=a)
        perf.append([a, score])

    return {
        'perf': perf,
        'pid': os.getpid(),
    }

if __name__ == '__main__':
    args = get_args()
    arch2performance = {}

    progress = tqdm(total=args.search_init_pop + args.search_mutate_epoch * args.search_mutate_number)

    progress.set_description('initial')

    edgeop, nodeop = get_edge_node_op(PRIMITIVES, args.search_space)
    archs = sample_valid_archs(args.search_layer, edgeop, nodeop, args.search_init_pop, PRIMITIVES)

    archs_passed = [[]]
    for a in archs:
        if len(archs_passed[-1]) == args.search_chunk:
            archs_passed.append([])
        archs_passed[-1].append(a)

    for a in archs_passed:
        res = eval_param_archs(args=args, archs=a)
        for line in res['perf']:
            arch2performance[str(line[0])] = line[1]

    for i in range(args.search_mutate_epoch):
        progress.set_description(f'epoch: {i}')

        # mutate architectures
        current_archs = list(arch2performance.items())
        current_archs = sorted(current_archs, key=lambda x:-x[1])
        mutated = current_archs[:args.search_mutate_number]
        arch_new = [[]]
        for arch in mutated:
            arch = eval(arch[0])
            a = mutate_arch(arch, edgeop, nodeop, PRIMITIVES)
            while str(a) in arch2performance: a = mutate_arch(arch, edgeop, nodeop, PRIMITIVES)
            if len(arch_new[-1]) == args.search_chunk: arch_new.append([])
            arch_new[-1].append(a)

        for a in arch_new:
            res = eval_param_archs(args=args, archs=a)
            for line in res['perf']:
                arch2performance[str(line[0])] = line[1]
    
    # derive final lists
    archs = sorted(list(arch2performance.items()), key=lambda x:-x[1])
    for x in archs:
        print(x)
    print([[eval(x[0]), x[1]] for x in archs])
    print(os.path.join(args.search_output, 'performance.dict'))
    torch.save([[eval(x[0]), x[1]] for x in archs], os.path.join(args.search_output, 'performance.dict'))
