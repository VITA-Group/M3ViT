import os
from datetime import datetime
import subprocess
import json


# submit the job to the cluster
def eval_edges(save_dir, exp_name, dataset='PascalContext', num_req_files=None):

    if dataset == 'PascalContext':
        database = 'PASCALContext'
        if num_req_files is None:
            num_req_files = 5105
    elif dataset == 'NYUD':
        database = 'NYUD'
        if num_req_files is None:
            num_req_files = 654
    else:
        raise ValueError('unknown database')

    chk_dir = os.path.join(save_dir, 'edges')
    if not os.path.exists(chk_dir):
        print('Experiment {} is not ready yet. Evaluation aborted.')
        return

    # check for filenames
    fnames = sorted(os.listdir(chk_dir))
    #if len(fnames) != num_req_files:
    if False:
        print('Number of files is incorrect')
    else:
        # seism path
        seism_cluster_dir = '/home/seism'

        # rsync to seism
        rsync_str = 'rsync -aP {}/ '.format(chk_dir)
        rsync_str += '{} '.format(os.path.join(seism_cluster_dir, 'datasets', database, exp_name))
        rsync_str += '--exclude=models --exclude=*.txt'
        print(rsync_str)
        os.system(rsync_str)

        # submit the job

        # subm_job_str = 'export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf; '
        subm_job_str = 'cp {}/parameters/HED.txt {}/parameters/{}.txt && ' \
                       .format(seism_cluster_dir, seism_cluster_dir, exp_name)
        # subm_job_str += 'python {}/src/scripts/eval_in_cluster.py {} {} read_one_cont_png fb 1 102 val' \
        subm_job_str += 'sbatch -J evalFb --array 1-102 {}/src/scripts/eval_in_cluster.py {} {} read_one_cont_png fb 1 102 val' \
            .format(seism_cluster_dir, exp_name, database)
        print(subm_job_str)
        os.system(subm_job_str)


# after cluster job is done, parse the output to generate .json file
def generate_pr_curves(save_dir, exp_name, dataset='PascalContext'):
    if dataset == 'PascalContext':
        database = 'PASCALContext'
    elif dataset == 'NYUD':
        database = 'NYUD'
    else:
        raise ValueError('unknown database')
    seism_cluster_dir = '/home/seism'

    timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
    chk_file = os.path.join(save_dir, 'edge_{}.json'.format(exp_name))

    os.chdir(seism_cluster_dir)
    command_to_run = ['matlab', '-nodesktop', '-nodisplay', '-nosplash', '-r', "install;pr_curves_to_file('"+database+"','val','"+exp_name+"');exit"]
    subprocess.run(command_to_run, check=True)
    eval_results = {}
    for measure in {'ods_f', 'ois_f', 'ap'}:
        tmp_fname = os.path.join(seism_cluster_dir, 'results', 'pr_curves', database,
                                 database + '_val_fb_' + exp_name + '_' + measure + '.txt')
        with open(tmp_fname, 'r') as f:
            eval_results[measure] = float(f.read().strip())
    try:
        with open(chk_file, 'r') as f:
            data = json.load(f)
        data.update(eval_results)
        eval_results = data
    except FileNotFoundError:
        pass
    with open(chk_file, 'w') as f:
        json.dump(eval_results, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', type=str, help='Run eval_edges or generate_pr_curves')
    parser.add_argument('--exp_names', type=str, help='Experiment names separated by comma. e.g. exp1 or exp1,exp2 ')
    parser.add_argument('--dataset', type=str, help='Name of dataset')
    opts = parser.parse_args()

    assert opts.dataset in ['PascalContext', 'NYUD']

    exp_names = opts.exp_names.split(',')
    for exp_name in exp_names:
        save_dir = '/home/{}'.format(exp_name)
        if opts.script == 'eval_edges':
            eval_edges(save_dir, exp_name, opts.dataset)
        elif opts.script == 'generate_pr_curves':
            generate_pr_curves(save_dir, exp_name, opts.dataset)
        else:
            raise ValueError('Select either eval_edges or generate_pr_curves')
