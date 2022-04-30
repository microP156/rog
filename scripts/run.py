import subprocess
import sys
import time
import datetime
import atexit
import os
import argparse

parser = argparse.ArgumentParser(description='ROG experiments')
parser.add_argument('-l', '--library', metavar='SSP', type=str, default='SSP',
                    choices=['SSP', 'FLOWN', 'ROG', 'BSP'], help='Type of experiment.')
parser.add_argument('-t', '--threshold', metavar='4', type=int, default=2,
                    help='Type of experiment.')
parser.add_argument('-c', '--control', metavar='indoors', default='',
                    choices=['indoors', 'outdoors', ''], help='Add Tc control.')
parser.add_argument('--no-compression', default=False,
                    action='store_true', help='Whether to use DEFSGDM compression.')
parser.add_argument('-e', '--epoch', default=4, type=int)
parser.add_argument('-n', '--note', default="", type=str, help="Note for exp")
args = parser.parse_args()

hosts = [
    '10.42.0.1',
    '10.42.0.2',
    # '10.42.0.6',
    '10.42.0.7',
    '10.42.0.8',
    # '10.42.0.9',
    '10.42.0.3',
    # '10.42.0.10'
]
leader_ip = '10.42.0.1'
# nic_names = ['wls1', 'wlx0013ef6f0c49', 'wlan0', 'wlan0', 'wlan0']
nic_names = ['wls1', 'wlx0013ef6f0c49', 'wlan0', 'wlan0', 'wlx0013ef5f09a3']
hosts = [f'user@{host}' for host in hosts]
hosts_set = list(sorted(set(hosts), key=hosts.index))  # remove duplicate but keep the order

# nic_names = ['wls1', 'wlx0013ef6f0c49', 'wlan0']
# fix_computation_time=2.0
# debug using ethernet
# leader_ip = '192.168.1.2'
# nic_names = ['eno1', 'eno1', 'enp0s31f6', 'enp0s31f6']

subps = []  # all subprocesses started by me


def cleanup():
    print('cleaning')
    for host in hosts_set:
        exec_ssh(host, 'docker stop -t 1 multi_robot_rl', block=True)
    for p in subps:
        tmp = ' '
        print(f'killing "{tmp.join(p.args)}"')
        p.kill()


atexit.register(cleanup)


def Popen(*args, **kwargs):
    p = subprocess.Popen(*args, **kwargs)
    subps.append(p)
    return p


def exec_local(command, block=False):
    args = ['bash', '-c', command]
    print(f'executing on local: {command}')
    p = Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if block:
        ret = p.wait()
        assert ret == 0, 'something wrong during excution: \n{p.stderr.readlines()}'
    return p


def exec_ssh(host, command, block=False):
    args = [
        'ssh', '-o', 'ServerAliveInterval 60', '-o', 'ServerAliveCountMax 120',
        host, command,
    ]
    print(f"executing on {host}: {command}")
    p = Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if block:
        ret = p.wait()
        assert ret == 0, f'something wrong during execution: \n{p.stderr.readlines()}'
    return p


def exec_docker(host, command, block=False):
    command = f'docker exec -t multi_robot_rl bash -c "{command}"'
    return exec_ssh(host, command, block=block)


print('restarting all containers')
for host in hosts_set:
    exec_ssh(host, 'docker restart -t 1 multi_robot_rl', block=True)

# E = [25, 3, 1, 8, 8]            # local updates for each worker
# batch_size = [4, 2, 4, 4, 4]
# idx_num = [500, 20, 20, 100, 100]    # dataset size for each worker
# 6, 48, 48, 20
# 4, 32, 32, 16
# 2, 24, 24, 8
E = [25, 1, 1, 1, 1]            # local updates for each worker
batch_size = [4, 2, 24, 24, 8]
idx_num = [500, 4, 4, 4, 4]    # dataset size for each worker
library = args.library
threshold = args.threshold
tc_control = args.control
epoch = args.epoch
compression_arg = '--compression-enable'
if args.no_compression:
    compression_arg = ''

if library == "BSP":
    library_arg = '--' + 'SSP' + '-enable'
    assert threshold == 0
else:
    library_arg = '--' + library + '-enable'

time_mark = datetime.datetime.now().strftime('%m-%d-%H-%M')
result_dir = f'result/{time_mark}-{library}-{threshold}-{tc_control}'
log_dir = f'{result_dir}/log'
chkpt_dir = f'{result_dir}/chkpt'
chkpt_worker_idx = 1
os.makedirs(log_dir)
os.makedirs(chkpt_dir)
with open(f'{result_dir}/config', 'w') as f:
    config_string = f'{library}:\nworld_size: {len(hosts)}\n' + \
        f'threshold: {threshold}\nbw control: {tc_control}\nepoch: {epoch}\nE: {E}\n' + \
        f'library_arg: {library_arg}\nnote: {args.note}'
    f.write(config_string)

server_log = None
for idx, host in enumerate(hosts):
    # MASTER_PORT={46666} MASTER_ADDR={leader_ip}
    exec_docker(host, f'tc qdisc del dev {nic_names[idx]} root')
    exec_docker(host, f'cd /home/work/adapt_noise && \
        export GLOO_SOCKET_IFNAME={nic_names[idx]} && \
        python3 -u adapt_noise_ssp.py --noise-type image_blur --epochs {epoch} -b {batch_size[idx]} -E {E[idx]} --idx-start {sum(idx_num[1:idx])} --idx-num {idx_num[idx]} --chkpt-dir {chkpt_dir} --chkpt-rank {chkpt_worker_idx} --world-size {len(hosts)} --wnic-name {nic_names[idx]} --rank {idx} --dist-url tcp://{leader_ip}:46666 --threshold={threshold} {library_arg} {compression_arg}>> {log_dir}/worker_{idx}.log 2>&1')
    if idx == 0:
        server_log = f'{log_dir}/worker_{idx}.log'
    # exec_docker(host, 'python3 -m pip install bitarray')
    # exec_docker(host, f'apt install -y iw >> {log_dir}/{idx} 2>&1')
    time.sleep(0.5)

# bw_record = 'bw_records/lab-30min.txt'
if len(tc_control) > 0:
    bw_record = f'bw_records/{tc_control}.txt'
    print('starting bandwidth replay')
    for idx, host in enumerate(hosts):
        if idx == 0:
            exec_docker(hosts[idx], f'cd /home/work && bash ./limit_bandwidth.sh {nic_names[idx]}')
            continue
        mode = 'leader' if idx == 0 else 'worker'
        exec_docker(hosts[idx], f'cd /home/work && python3 -u replay_bandwidth.py {bw_record} {mode} {idx+2} > adapt_noise/{log_dir}/bw_replay-{idx}.txt 2>&1')
else:
    for idx, host in enumerate(hosts):
        exec_docker(hosts[idx], f'cd /home/work && bash ./limit_bandwidth.sh {nic_names[idx]}')

for idx, host in enumerate(hosts):
    exec_docker(host, f'cd /home/work && python3 -u record_energy.py 0.5 >adapt_noise/{log_dir}/energy-{idx}.txt 2>&1')

# if noise == 'Y':
#     for idx, host in enumerate(hosts[1:]):
#         exec_local(f'bash iperf_noise.sh {leader_ip} {host} {datasize} {sleep} >> {log_dir}/noise_{idx+1}.log 2>&1')

print('all started')
print('when enough, press ctrl+c ONCE, then wait for me to kill all started processes')
print('but youd better double check then')
while True:
    with open(server_log, "rb") as file:
        try:
            file.seek(-2, os.SEEK_END)
            while file.read(1) != b'\n':
                file.seek(-2, os.SEEK_CUR)
        except OSError:
            file.seek(0)
        last_line = file.readline().decode()
    if last_line.startswith("Whole thread terminated") or last_line.startswith('BrokenPipeError'):
        print('Server down.')
        cleanup()
        break
    else:
        time.sleep(5)
# for p in subps:
#     p.wait()
