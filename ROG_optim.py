import torch
from .new_ROG_utils import ROG_Local_Worker, ROG_Parameter_Server, layer_unit
from .DEFSGDM.DEFSGDM import DEFSGDM_server, DEFSGDM_worker

server = 0
worker = 1

class ROG_Worker(torch.optim.Optimizer):
    def __init__(self, params, args, *_args, cfg=None, **_kwargs):
        super().__init__(params, {})
        if args.rank == 0:
            self.role = server
        else:
            self.role = worker
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.layer_info=[]
        self.parameters = []
        start_idx=0
        start_pos=0
        self.temp_optimizer = None
        self.model_numel = 0
        for group in self.param_groups:
            for p in group['params']:
                each_layer_rows,start_idx,start_pos = layer_unit(p, start_idx, start_pos)
                self.parameters.append(p)
                self.model_numel += p.numel()
                self.layer_info.append(each_layer_rows)

        if self.role == server:
            self.optimizer = DEFSGDM_server(params=params, worker_num=args.world_size, device=self.device, local_copy=False)
            self.rog_server = ROG_Parameter_Server(args, parameters=self.parameters, layer_info=self.layer_info, optimizer=self.optimizer, device=self.device)

            self.rog_server.start()

        else:
            self.optimizer = DEFSGDM_worker(params=params, *_args, device=self.device, **_kwargs)
            self.rog_worker = ROG_Local_Worker(args, parameters=self.parameters, layer_info=self.layer_info, model_numel=self.model_numel, optimizer=self.optimizer, device=self.device)
        print("worker start\n")

    def init(self):
        self.optimizer.init_state()

    def set_optimizer(self, params, *args, **kwargs):
        assert self.role == worker
        self.temp_optimizer = DEFSGDM_worker(params, *args, device=self.device, **kwargs)
        return self.temp_optimizer

    def unset_optimizer(self):
        assert self.role == worker
        self.temp_optimizer = None

    def step(self):
        assert self.role == worker
        if self.temp_optimizer:
            self.temp_optimizer.step()
        else:
            self.optimizer.step()

    def push_and_pull(self):
        assert self.role == worker
        # if self.temp_optimizer:
        #     self.temp_optimizer.retrieve_state(self.optimizer)
        _, transmission_time = self.rog_worker.push_update()
        self.rog_worker.pull_model(transmission_time)

    def zero_grad(self):
        self.optimizer.zero_grad()
