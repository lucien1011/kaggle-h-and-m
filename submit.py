
slurm_dir = 'storage/slurm/'

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmd',action='store')
    parser.add_argument('--time',action='store',default='04:00:00')
    parser.add_argument('--mem',action='store',default='32gb')
    parser.add_argument('--gpu',action='store_true')
    return parser.parse_args()

def submit(args):
    import datetime
    import os
    from slurm import SLURMWorker

    name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    script_file_name = os.path.join(slurm_dir,name+'.slurm')
    base_dir = os.environ['BASE_PATH']

    worker = SLURMWorker()
    slurm_commands = """
cd {base_path}
source setup_hpg.sh
{commands}
""".format(commands=args.cmd,base_path=base_dir)
    worker.make_sbatch_script(
            script_file_name,
            name,
            'kin.ho.lo@cern.ch',
            1,
            args.mem,
            args.time,
            slurm_dir,
            slurm_commands,
            gpu=args.gpu,
            )
    worker.sbatch_submit(script_file_name)

if __name__ == '__main__':
    args = parse_arguments()
    submit(args)

