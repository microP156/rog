import os

# from torch import tensor
# from torch.distributed.distributed_c10d import recv
import socket
import queue

import threading
import pickle
import multiprocessing as mp
import time
from utils import AverageMeter, accuracy, mkdir_p, savefig
import torch
# from math import cos, pi
import torch.distributed as dist
import copy
from email.policy import default
import sys
from collections import defaultdict
import subprocess
from DEFSGDM.DEFSGDM import compress_cost, decompress_cost
import numpy as np
from read_bandwidth import get_bandwidth
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
# from tqdm import tqdm
# import math
MAX_RECV_SIZE=4*1024

class TCPMessageStream:
    NUM_SIZE=4
    def __init__(self,sock:socket.socket):
        self.sock=sock
        self.send_queue=queue.Queue()
        def send_task():
            while True:
                msg=self.send_queue.get()
                msize = len(msg)
                self.sock.send(msize.to_bytes(self.NUM_SIZE,"big")+msg)
        self.send_thread=threading.Thread(target=send_task, daemon=True)
        self.send_thread.start()

        self.recv_queue=queue.Queue()
        def recv_task():
            buffer=bytearray()
            while True:
                while len(buffer) < self.NUM_SIZE:
                    buffer +=self.sock.recv(MAX_RECV_SIZE)
                msize=int.from_bytes(buffer[:self.NUM_SIZE],"big")
                buffer=buffer[self.NUM_SIZE:]
                while len(buffer)<msize:
                    buffer +=self.sock.recv(MAX_RECV_SIZE)
                msg =buffer[:msize]
                buffer=buffer[msize:]
                self.recv_queue.put(msg)
        self.recv_thread=threading.Thread(target=recv_task, daemon=True)
        self.recv_thread.start()

    def send(self,msg):
        self.send_queue.put(msg)

    def recv(self):
        return self.recv_queue.get()

def freeze_bn(m):
    if isinstance(m, torch.nn.LayerNorm) or isinstance(m, torch.nn.BatchNorm2d):
        m.training = False
        m.track_running_stats = True

class Parameter_Server:
    def __init__(self, args, ps_ip, ps_port, world_size, threshold,
            model, optimizer, communication_library, device,
            batch_size, chkpt_dir, FLOWN_enable=False, compression_enable=True):
        self.world_size = world_size
        self.threshold = threshold
        self.model = model
        self.gathered_weight = mp.Queue(maxsize=1000)
        self.lock = threading.Lock()
        self.training_step=[0 for _ in range(world_size)]
        self.stall_time = [AverageMeter() for _ in range(world_size)]
        self.stall_num = [0 for _ in range(world_size)]
        self.min_step = [mp.Queue(maxsize=1) for i in range(world_size)]
        self.communication_library=communication_library
        self.device = device
        self.FLOWN_enable = FLOWN_enable
        self.COMPRESSION = compression_enable
        if self.COMPRESSION:
            assert self.communication_library == 'tcp'
        self.updated = True
        self.recved = True
        self.optimizer = optimizer
        logging.info(f"Threshold {threshold}")
        logging.info(f"FLOWN_enabled {FLOWN_enable}")
        logging.info(f"COMPRESSION_enabled {compression_enable}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((ps_ip, ps_port))
        sock.listen(world_size)
        self.chkpt_dir = chkpt_dir

        self._stop_event = threading.Event()
        proc = []
        if self.COMPRESSION:
            t = threading.Thread(target=self.checkpoint_per_period, daemon=True)
            proc.append(t)
        self.isupdated = []
        # [[computation & compression time, sleep time, batchsize] * world_size]
        self.sleep_time_lock = threading.Lock()
        self.computation_compression = np.zeros([world_size, 3])
        for _ in range(world_size):
            self.isupdated.append(mp.Queue(maxsize=10))
        if not self.COMPRESSION:
            t = threading.Thread(target=self.parameter_server_optimizer, args=(optimizer,), daemon=True)
            proc.append(t)

        self.client_addresses = []
        self.worker_bw = np.array([0.]*8)
        client_streams = []
        for i in range(world_size):
            client_sock, client_address = sock.accept()
            client_stream = TCPMessageStream(client_sock)
            t = threading.Thread(target=self.each_parameter_server,args=(client_stream,client_address,i), daemon=True)
            proc.append(t)
            self.client_addresses.append(client_address[0])
            client_streams.append(client_stream)
        logging.info(f'client address {self.client_addresses}')
        for i, stream in enumerate(client_streams):
            stream.send(pickle.dumps(i))

        if self.FLOWN_enable:
            self.query_idx = []
            self.flown_query = mp.Queue(maxsize=10000)
            # Use queue to block out excessive request
            self.flown_next = [mp.Queue(maxsize=2) for _ in range(world_size)]
            self.transmitting_idxes = mp.Queue(maxsize=2)
            proc.append(threading.Thread(target=self.flown_control, daemon=True))

        for t in proc:
            t.start()
        logging.info("start parameter server")
        for t in proc:
            t.join()
        sock.close()

    def get_rates(self, wnic='wls1'):
    #"""
    #Returns a dict() with str(ip address) as keys and float(bandwidth) (Mbps) as values.
    #"""
        def find_ip_by_mac(mac):
            sp = subprocess.Popen(f'ip neighbor | grep {mac} | cut -d" " -f1 | grep 10.42', shell=True, stdout=subprocess.PIPE)
            ip = sp.stdout.read().decode().strip()
            # print(f"ip {ip}")
            return ip

        sp = subprocess.Popen(f'iw dev {wnic} station dump', shell=True, stdout=subprocess.PIPE)
        station_dump = sp.stdout.read().decode()

        bw_dict = dict()
        for line in station_dump.split('\n'):
            if 'Station' in line:
                current_station = line.split()[1]
                current_ip = find_ip_by_mac(current_station)
            if 'tx bitrate' in line:
                rate = float(line.split()[2])
                bw_dict[current_ip] = rate
        # print(f"bw_dict {bw_dict}")
        return bw_dict

    def suggest_sleep_time(self, l, rank):
        # first tune the batch size of the slow devices (small batchszie)
        # to take a bit less time than fast devices (large batchsize) to finish computation and compression,
        # then use this function to make the slow devices the sleep for certain time
        # computation_compression: [[compute&compress time, sleep time, batchsize]]
        self.sleep_time_lock.acquire()
        arr = self.computation_compression
        arr[rank] = np.array(l)
        # if some workers have not sent its batchsize
        if np.any(self.computation_compression[:, 2] <= 0.):
            self.sleep_time_lock.release()
            return
        # largest batchsize
        fast = arr.argmax(axis=0)[2]
        # same batchsize can appear
        all_fast = np.where(arr[:, 2] == arr[fast, 2])[0]
        # find the one with largest compute&compress time
        fast = all_fast[np.argmax(arr[:, 0][all_fast])]
        if arr[fast, 0] < np.max(arr[:, 0]):
            logging.info(f'Warning: slow devices are still taking longer time to compute and compress;\narray {arr}')
            slowest = np.argmax(arr[:, 0])
            arr[:, 1] = arr[slowest, 0] - arr[:, 0]
        else:
            arr[:, 1] = arr[fast, 0] - arr[:, 0]
        self.sleep_time_lock.release()

    def flown_control(self):
        while not self._stop_event.is_set():
            while not self.flown_query.empty() or len(self.query_idx) == 0:
                self.query_idx.append(self.flown_query.get())
            # choose two fastest
            for _ in range(2):
                next_idx = -1
                logging.info(f"current query_idx {self.query_idx} worker bw {self.worker_bw}")
                if len(self.query_idx) == 0:
                    break
                for i in self.query_idx:
                    if self.training_step[i] <= max(self.training_step) - self.threshold:
                        next_idx = i
                        break
                # descending
                if next_idx == -1:
                    sorted_bw = np.argsort(self.worker_bw)[::-1]
                    for idx in sorted_bw:
                        if idx in self.query_idx:
                            next_idx = idx
                            break
                assert next_idx != -1
                # only when put is successful will the following put be executed
                self.transmitting_idxes.put(1)
                for i in range(self.world_size):
                    while True:
                        try:
                            self.flown_next[i].put(next_idx)
                            break
                        except:
                            continue
                if next_idx in self.query_idx:
                    self.query_idx.remove(next_idx)
                else:
                    logging.info('Query_idx: It happens again')
                logging.info(f'Next_idx {next_idx}')
            # print(f"query_idx {self.query_idx}; training_step {self.training_step}")
        return

    def each_parameter_server(self, client_stream: TCPMessageStream, client_address, rank):
        while True:
            msg = pickle.loads(client_stream.recv())
            if self._stop_event.is_set():
                msg = 'terminate'
            if msg == 'ask':
                if self.FLOWN_enable:
                    logging.info(f"{rank} ask for model")
                    self.flown_query.put(rank)
                    idx = self.flown_next[rank].get()
                    # print(f"Ask {rank} get {idx}")
                    while idx != rank:
                        idx = self.flown_next[rank].get()
                        # print(f"Ask {rank} get {idx}")
                # print(f'Sending parameters to client {rank}')
                if self.communication_library == "gloo":
                    srcrank = pickle.loads(client_stream.recv())
                    for p in self.model.parameters():
                        dist.send(p, srcrank)
                if self.communication_library == "tcp":
                    # print(f'Sending parameters to client {rank}')
                    if self.COMPRESSION:
                        client_stream.send(pickle.dumps(
                            (self.optimizer.send(rank), self.computation_compression[rank, 1])))
                        self.optimizer.update_error(rank)
                    else:
                        client_stream.send(pickle.dumps(self.model))
                if self.FLOWN_enable:
                    self.transmitting_idxes.get()
                # print(f'Finish sending parameters to client {rank}')
            if msg == 'send':
                logging.info(f'Recv parameters from client {rank}')
                if self.communication_library == "gloo":
                    lr = pickle.loads(client_stream.recv())
                    weight = []
                    for p in self.model.parameters():
                        recv_tensor=torch.ones_like(p)
                        dist.recv(recv_tensor,srcrank)
                        weight.append(recv_tensor)
                if self.communication_library == "tcp":
                    if self.COMPRESSION:
                        (sign_model, param_group), computation_compression, self.worker_bw[rank] = pickle.loads(client_stream.recv())
                        self.optimizer.recv(sign_model, param_group, rank)
                        self.suggest_sleep_time(computation_compression, rank)
                    else:
                        weight,lr=pickle.loads(client_stream.recv())
                if not self.COMPRESSION:
                    self.gathered_weight.put((weight, lr, rank))

                stime = time.time()
                with self.lock:
                    logging.info(f'{rank}, {self.training_step}, before; compress {compress_cost.avg:.2E}; decompress {decompress_cost.avg:.2E} stall time avg:, {[round(t.avg,2) for t in self.stall_time]}, {round(sum([t.avg for t in self.stall_time])/len(self.stall_time),2)}, sum:, {[round(t.sum,2) for t in self.stall_time]}, stall num, {self.stall_num}')
                    before = min(self.training_step)
                    self.training_step[rank] += 1
                    now = min(self.training_step)
                    logging.info(f'{rank}, {self.training_step}, after')
                if not self.COMPRESSION:
                    msg = self.isupdated[rank].get()
                    assert msg == "ok"
                if self.FLOWN_enable:
                    self.flown_query.put(rank)
                    idx = self.flown_next[rank].get()
                    # print(f"Recv {rank} get {idx}")
                    while idx != rank:
                        idx = self.flown_next[rank].get()
                        # print(f"Recv {rank} get {idx}")
                # min_step = min(self.training_step)
                else:
                    if before != now:
                        for i in range(self.world_size):
                            if not self.min_step[i].empty():
                                try:
                                    self.min_step[i].get_nowait()
                                except queue.Empty:
                                    pass
                            if i == rank:
                                continue
                            self.min_step[i].put(now)
                    else:
                        while self.training_step[rank] > now + self.threshold:
                            now = self.min_step[rank].get()
                client_stream.send(pickle.dumps("ok"))
                etime = time.time()
                if etime - stime > 0.05:
                    self.stall_num[rank] += 1
                self.stall_time[rank].update(etime - stime)
                if etime - stime < 1e-2:
                    logging.info(f'{rank}, stall waiting, 0.0')
                else:
                    logging.info(f'{rank}, stall waiting, {etime - stime}')
                if self.FLOWN_enable:
                    self.transmitting_idxes.get()
                self.recved = True
                # print(f'Fine Recv parameters from client {rank}')
            if msg == "finish":
                logging.info(f'{rank} finish one epoch.')
                client_stream.send(pickle.dumps("ok"))
            if msg == "terminate":
                if not self.COMPRESSION:
                    self.gathered_weight.put((None,None,None))
                self._stop_event.set()
                logging.info(f"server for {rank} terminated")
                break

    def parameter_server_optimizer(self,optimizer):
        terminated = 0
        while True:
            weight,lr,rank=self.gathered_weight.get()
            if weight is None and lr is None and rank is None:
                terminated += 1
                if terminated < self.world_size:
                    continue
                else:
                    logging.info('Optimizer terminated')
                    break
                logging.info(f'Terminated {terminated} world_size {self.world_size}')
            for p,g in zip(self.model.parameters(),weight):
                p.grad=g/self.world_size
            if lr is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad()
            self.isupdated[rank].put("ok")
            self.updated = True

    
    def checkpoint_per_period(self, period=60, decay_step=20):
        step = 1
        start = time.time()
        while True:
            if self._stop_event.is_set():
                return
            if not self.recved:
                time.sleep(5)
                continue
            else:
                self.recved = False
            if self.COMPRESSION:
                self.optimizer.step()
            torch.save(self.model.state_dict(), f'{self.chkpt_dir}/{time.time() - start:8.2f}-{max(self.training_step)}.chkpt')
            sleep_time = period - (time.time() - start) % period
            time.sleep(sleep_time)
            step += 1
            if step % decay_step == 0:
                period *= 3

class Local_Worker:
    def __init__(self, args, model, ps_ip, ps_port, criterion,
        communication_library, train_loader, test_loader, device, lr_scheduler, optimizer, mixup_fn, compression_enable=True):

        self.args = args
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.communication_library = communication_library
        self.device = device
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.mixup_fn = mixup_fn
        self.optimizer = optimizer
        self.COMPRESSION = compression_enable
        self.t_communication = AverageMeter()
        self.t_computation = AverageMeter()
        self.batchsize = 0
        self.sleep_time = 0.
        self.lock = threading.Lock()
        self.versions = []
        self.losses = AverageMeter()
        if self.COMPRESSION:
            assert self.communication_library == 'tcp'
        logging.info(f"Device {device}; cudnn.benchmark={torch.backends.cudnn.benchmark}; cudnn.enabled={torch.backends.cudnn.enabled}")
        # Warm up phase
        # self.train(0, 0, 0, 0.0001, local_update=args.E, warm_up=True, warm_up_iter=10)
        logging.info('Warm up complete')
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        time.sleep(1) # waiting for ps start
        sock.connect((ps_ip, ps_port))
        self.sock = TCPMessageStream(sock)
        logging.info("conneted to parameter server")
        msg = self.sock.recv()
        self.rank = pickle.loads(msg)
        logging.info(f"All workers ready, my rank is {self.rank}")
        logging.info(self.optimizer.defaults)

        self._stop_event = threading.Event()
        self.checkpoint_thread = None
        self.training_step = 0
        self.bandwidth = 100
        if args.FLOWN_enable:
            self.rb_thread = threading.Thread(target=self.read_bandwidth_thread, daemon=True)
            self.rb_thread.start()
        logging.info('Worker started.')
        # if args.chkpt_rank == args.rank:
        #     print("Checkpoint will be handled here.")
        #     self.checkpoint_thread = threading.Thread(target=self.checkpoint_per_period, args=(), daemon=True)
        #     self.checkpoint_thread.start()

    def read_bandwidth_thread(self):
        while True:
            if self._stop_event.is_set():
                break
            self.bandwidth = get_bandwidth(self.args.wnic_name)
            time.sleep(1)


    def pull_model(self):
        # print("start ask for new model",flush=True)
        start=time.time()
        self.sock.send(pickle.dumps("ask"))
        if self.communication_library=="gloo":
            self.sock.send(pickle.dumps(dist.get_rank()))
            for p in self.model.parameters():
                dist.recv(p,0)
        if self.communication_library=="tcp":
            if self.COMPRESSION:
                data = self.sock.recv()
            else:
                self.model=pickle.loads(self.sock.recv())
        end = time.time()
        if self.COMPRESSION:
            logging.info(f'Recv size {len(data) / 1024/1024:.2f} MB')
            (sign_model, param_group), self.sleep_time = pickle.loads(data)
            self.optimizer.decompress_optimize(sign_model, param_group)
        # print("complete",flush=True)
        return end-start

    def push_update(self, iteration):
        # print("start push update",flush=True)
        if self.COMPRESSION:
            data = self.optimizer.compress()
            self.optimizer.update_error()
        start=time.time()
        self.sock.send(pickle.dumps("send"))
        if self.communication_library=="gloo":
            self.sock.send(pickle.dumps(self.lr))
            for p in self.model.parameters():
                dist.send(p.grad.detach(), 0)
        if self.communication_library == "tcp":
            if self.COMPRESSION:
                t_compute_compress = self.t_computation.val + compress_cost.val + decompress_cost.val
                data = pickle.dumps((data, 
                                     [t_compute_compress, 0., self.batchsize],
                                     self.bandwidth))
                logging.info(f'Send size {len(data) / 1024/1024:.2f} MB')
                self.sock.send(data)
            else:
                weight=[]
                for p in self.model.parameters():
                    weight.append(p.grad.detach())
                self.sock.send(pickle.dumps((weight, self.lr)))
        msg = pickle.loads(self.sock.recv())
        assert msg == 'ok'
        end = time.time()

        # for p in self.model.parameters():
        #     p.grad = torch.zeros_like(p.grad)
        # print("complete",flush=True)
        return end - start

    def terminate(self):
        logging.info('Worker terminated.')
        self.sock.send(pickle.dumps("terminate"))
        self._stop_event.set()

    def train(self, epoch, start_time, num_updates, lr, local_update, warm_up=False, warm_up_iter=20):
        # bar = Bar('Processing', max=len(self.train_loader))
        start = time.time()
        top1 = AverageMeter()
        losses = self.losses
        self.model.train()
        self.model.apply(freeze_bn)
        self.lr = lr
        batchsize = self.train_loader.batch_size
        itered = 0
        iteration = 0
        all_iters = len(self.train_loader)
        loader = iter(self.train_loader)
        finish = False
        while itered < all_iters:
            self.lock.acquire()
            for _ in range(local_update):
                try:
                    images, targets = next(loader)
                except StopIteration:
                    finish = True
                    break
                itered += 1
                images = images.to(self.device)
                targets = targets.to(self.device)
                images, targets = self.mixup_fn(images, targets)
                output = self.model(images)
                loss = self.criterion(output, targets)
                loss.backward()
                losses.update(loss.item(), images.size(0))
            end = time.time()
            self.t_computation.update(end - start)
            if warm_up:
                self.lock.release()
                warm_up_iter -= 1
                if warm_up_iter <= 0:
                    self.t_communication.reset()
                    self.t_computation.reset()
                    compress_cost.reset()
                    decompress_cost.reset()
                    self.optimizer.init_state()
                    return 0., 0.
                else:
                    sign, param_groups = self.optimizer.compress()
                    self.optimizer.decompress_store(sign, param_groups)
                    logging.info(f"\nWarmup worker{self.rank} compress:{compress_cost.val:.2E} decompress:{decompress_cost.val:.2E}")
                    logging.info(f"batchsize:{local_update * batchsize} comp:{end - start:.2f} avg_comp:{self.t_computation.avg:.2f}")
                    start = time.time()
                    continue
            iteration += 1
            if self.sleep_time > 0.:
                time.sleep(self.sleep_time)
            self.training_step += 1
            push_update_time = self.push_update(epoch)
            pull_model_time = self.pull_model()
            for p in self.model.parameters():
                p.grad.zero_()
            self.batchsize = local_update * batchsize
            num_updates += 1
            self.lr_scheduler.step()
            self.lock.release()
            self.t_communication.update(pull_model_time + push_update_time)
            if self.training_step < 5 and self.sleep_time == 0.:
                logging.info('Start sleep.')
                self.sleep_time = 20 - (self.t_computation.val + compress_cost.val + decompress_cost.val)
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)
            logging.info(f"Iteration {self.training_step} worker{self.rank} comm:{pull_model_time + push_update_time:.2f}s avg_comm:{self.t_communication.avg:.2f}s compress:{compress_cost.avg:.2E} decompress:{decompress_cost.avg:.2E}")
            logging.info(f"batchsize: {local_update * batchsize} comp:{end - start:.2f}s avg_comp:{self.t_computation.avg:.2f}s sleep:{self.sleep_time:.2f} total compute+compress+sleep:{self.t_computation.val + compress_cost.val + decompress_cost.val + self.sleep_time:.2f}")
            logging.info(f"loss: {losses.val:.2f} loss avg: {losses.avg:.2f} scale avg send {self.optimizer.send_scale_monitor.avg:.2E} avg recv: {self.optimizer.recv_scale_monitor.avg:.2E} lr {self.lr_scheduler.get_last_lr()}\n")
            
            start = time.time()
            if itered > 0:
                self.lr = None
            if finish:
                break
        self.sock.send(pickle.dumps("finish"))
        msg = pickle.loads(self.sock.recv())
        assert msg == 'ok'
        return losses.avg, num_updates

    def validate(self, epoch,start_time):
        # switch to evaluate mode
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            correct = 0
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                _, top_id = output.max(1)
                correct += torch.sum(top_id == y)
            correct_rate = correct / (len(self.test_loader) * self.args.batchsize)
            print(f'{epoch} epoch accuracy: {correct_rate}')
        return correct_rate

    def checkpoint_per_period(self, period=120, decay_step=20):
        step = 1
        start = time.time()
        while True:
            if self._stop_event.is_set():
                return
            # with self.lock:
            torch.save(self.model.state_dict(), f'{self.args.chkpt_dir}/{time.time() - start:8.2f}-{self.training_step}.chkpt')
            sleep_time = period - (time.time() - start) % period    # sleep till next minute

            time.sleep(sleep_time)
            step += 1
            if step % decay_step == 0:
                period *= 2