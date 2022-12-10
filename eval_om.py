import os
import sys
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform

sys.path.append(os.getcwd())
from model.VAEProcessor import ModelProcessor
from data.dataset_h36m import DatasetH36M
from utils.logger import *

def get_multimodal_gt():
    all_data = []
    data_gen = dataset.iter_generator(step=params["t_his"])
    for data in data_gen:
        data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    all_start_pose = all_data[:, params["t_his"] - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < params["multimodal_threshold"])
        traj_gt_arr.append(all_data[ind][:, params["t_his"]:, :])
    return traj_gt_arr

def get_gt(data, t_his):
    gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    return gt[:, t_his:, :]

def compute_diversity(pred, gt):
    if pred.shape[0] == 1:
        return 0.0
    dist = pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist.mean().item()
    return diversity


def compute_ade(pred, gt):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2).mean(axis=1)
    return dist.min()


def compute_fde(pred, gt):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)[:, -1]
    return dist.min()

if __name__ == '__main__':
    print(f"Process ID: {os.getpid()}\n")
    # print(os.__file__)
    # algos= ["dlow", "vae"]
    algos = ["dlow"]
    
    # model_type = ""
    # model_type = "mix_precision"
    # model_type = "force_fp32"
    model_type = "force_fp16"

    params = {   
        "data": "test",
        "t_his": 25,
        "t_pred": 100,
        "multimodal_threshold": 0.5,
        "seed": 0,
        "concat_hist": False,
        "result_dir": "results",

        "model":
        {
            "vae_model_path": f"checkpoints/vae_0500_batch10_{model_type}.om",
            "dlow_model_path": f"checkpoints/dlow_0500_xandz_{model_type}.om",
            "use_dlow": True,
            "batch_size": 10,
            "num_seeds": 1,
            "nz": 128,
        }     
    }

    np.random.seed(params["seed"])

    os.makedirs(params["result_dir"], exist_ok=True)

    logger = create_logger(os.path.join(params["result_dir"], 'log_eval.txt'))

    """data"""
    dataset = DatasetH36M(params["data"], params["t_his"], params["t_pred"])
    # traj_gt_arr = get_multimodal_gt()

    """model"""
    processor = ModelProcessor(params["model"])

    """stats init"""
    stats_func = {'Diversity': compute_diversity, 'ADE': compute_ade,
                  'FDE': compute_fde}
    stats_names = list(stats_func.keys())
    stats_meter = {x: {y: AverageMeter() for y in algos} for x in stats_names}
    
    """inference"""
    data_gen = dataset.iter_generator(step=params["t_his"])
    num_samples = 0
    num_seeds = params["model"]["num_seeds"]

    seconds = 0
    for i, data in enumerate(data_gen):
        # if i > 49:
        #     break
        
        # data = np.load('data.npy')
        num_samples += 1
        gt = get_gt(data, params["t_his"])
        # gt_multi = traj_gt_arr[i]
        
        begin = time.time()
        for algo in algos:
            pred = processor.predict(data, params["t_his"], params["concat_hist"])
            if pred is None:
                print("!!!!! OM Model execution Failed !!!!!")
                exit()
            # np.save("pred_om.npy", pred)
            
            for stats in stats_names:
                val = 0
                for pred_i in pred:
                    val += stats_func[stats](pred_i, gt) / num_seeds
                stats_meter[stats][algo].update(val)
        seconds = seconds + (time.time() - begin)

        print('-' * 80)
        for stats in stats_names:
            str_stats = f'{num_samples:04d} {stats}: ' + ' '.join([f'{x}: {y.val:.4f}({y.avg:.4f})' for x, y in stats_meter[stats].items()])
            print(str_stats)
        # break

    logger.info('=' * 80)
    logger.info(f"Model type: {model_type}")
    for stats in stats_names:
        str_stats = f'Total {stats}: ' + ' '.join([f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()])
        logger.info(str_stats)
    logger.info(f"Latency = {seconds/num_samples} secs")
    logger.info('=' * 80)
    print(num_samples)
    print(seconds)
    

    # with open('%s/stats_%s.csv' % (cfg.result_dir, args.num_seeds), 'w') as csv_file:
    #     writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + algos)
    #     writer.writeheader()
    #     for stats, meter in stats_meter.items():
    #         new_meter = {x: y.avg for x, y in meter.items()}
    #         new_meter['Metric'] = stats
    #         writer.writerow(new_meter)


