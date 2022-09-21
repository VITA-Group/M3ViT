r"""
Noisy gate for gshard and switch
"""
from fmoe.gates.base_gate import BaseGate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math
import numpy as np
from collections import Counter
from pdb import set_trace

class NoisyGate_VMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, noise_std=1, no_noise=False,
                 return_decoupled_activation=False,regu_experts_fromtask=False,num_experts_pertask=-1,num_tasks=-1,
                 regu_sem=False,sem_force = False,regu_subimage=False):
        super().__init__(num_expert, world_size)
        self.w_gate = nn.Parameter(
            torch.zeros(d_model, self.tot_expert), requires_grad=True
        )

        self.return_decoupled_activation = return_decoupled_activation
        if self.return_decoupled_activation:
            self.w_gate_aux = nn.Parameter(
                torch.zeros(d_model, self.tot_expert), requires_grad=True
            )

        self.top_k = top_k
        self.no_noise = no_noise
        self.noise_std = noise_std

        self.softmax = nn.Softmax(1)

        self.activation = None
        self.select_idx = None
        self.regu_experts_fromtask= regu_experts_fromtask
        self.num_experts_pertask = num_experts_pertask
        self.num_tasks = num_tasks
        self.regu_sem = regu_sem
        self.regu_subimage = regu_subimage
        self.patch_size = 16
        
        if self.regu_sem:
            from losses.loss_functions import SoftMaxwithLoss
            self.criterion = SoftMaxwithLoss()
            self.num_class = 40
            self.head = nn.Linear(num_expert, self.num_class)
            self.semregu_loss = 0.0
        if self.regu_subimage:
            self.regu_subimage_loss = 0.0
            self.subimage_tokens = 5
        if self.regu_experts_fromtask:
            self.start_experts_id=[]
            start_id = 0
            for i in range(self.num_tasks):
                start_id = start_id + int(i* (self.tot_expert-self.num_experts_pertask)/(self.num_tasks-1))
                self.start_experts_id.append(start_id)
            print('self.start_experts_id',self.start_experts_id)
        self.reset_parameters()

    def reset_parameters(self):
        # Approach is the same as in torch.nn.Linear
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L88

        torch.nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))

        if self.return_decoupled_activation:
            torch.nn.init.kaiming_uniform_(self.w_gate_aux, a=math.sqrt(5))

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.top_k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(
            torch.tensor([0.0], device=clean_values.device),
            torch.tensor([1.0], device=clean_values.device),
        )

        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_expert = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def set_loss(self, loss):
        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss
    def get_semregu_loss(self):
        return self.semregu_loss

    def get_regu_subimage_loss(self):
        return self.regu_subimage_loss


    # def get_groundtruth_sem(self, sem):
    #     batch = sem.shape[0]
    #     hint = np.ones(batch,1,int(sem.shape[2]/self.patch_size),int(sem.shape[3]/self.patch_size))*255
    #     idx = 0
    #     for k in range(batch):
    #         for i in range(int(sem.shape[2]/self.patch_size)):
    #             for j in range(int(sem.shape[3]/self.patch_size)):
    #                 patch = sem[k][:,self.patch_size*i:self.patch_size*(i+1),self.patch_size*j:self.patch_size*(j+1)].cpu().numpy().flatten()
    #                 index , num=Counter(patch).most_common(1)[0]
    #                 if num>0.4*(self.patch_size*self.patch_size):
    #                     hint[k,:,i,j]=index
    #                     if index != 255:
    #                         idx = idx+1
    #     print(idx/(batch*int(sem.shape[2]/self.patch_size)*int(sem.shape[3]/self.patch_size)),'percent token will be used')        
    #     return torch.tensor(hint, device=sem.device) 
     
    def forward(self, inp, task_id=None,sem=None):
        shape_input = list(inp.shape)
        # print(shape_input)
        channel = shape_input[-1]
        other_dim = shape_input[:-1]
        inp = inp.reshape(-1, channel)

        if self.regu_experts_fromtask and (task_id is not None):
            clean_logits = inp @ self.w_gate[:,self.start_experts_id[task_id]:self.start_experts_id[task_id]+self.num_experts_pertask]
            raw_noise_stddev = self.noise_std / self.num_experts_pertask
        else:
            clean_logits = inp @ self.w_gate
            raw_noise_stddev = self.noise_std / self.tot_expert
        noise_stddev = raw_noise_stddev * self.training

        if self.regu_sem and (sem is not None):
            batch = sem.shape[0]
            prior_selection = clean_logits.reshape(batch,-1,self.num_expert)[:,1:,:]
            prior_selection = prior_selection.reshape(-1,self.num_expert)
            prior_out = self.head(prior_selection)
            prior_out = prior_out.reshape(batch,sem.shape[2],sem.shape[3],self.num_class)
            prior_out = prior_out.permute(0,3,1,2)
            # hint = self.get_groundtruth_sem(sem)
            semregu_loss = self.criterion(prior_out,sem)
            # print('during forward regu loss',semregu_loss)
            self.semregu_loss = semregu_loss
            # print('clean_logits',clean_logits.shape,sem.shape)

        if self.regu_subimage and (sem is not None):
            self.regu_subimage_loss = 0
            batch_size = sem.shape[0]
            prior_selection = clean_logits.reshape(batch_size,-1,self.num_expert)[:,1:,:]
            prior_selection = prior_selection.reshape(batch_size,30,40,self.num_expert)
            for k in range(batch_size):
                for i in range(int(30/self.subimage_tokens)):
                    for j in range(int(40/self.subimage_tokens)):
                        subimage_selection = prior_selection[k,self.subimage_tokens*i:self.subimage_tokens*(i+1),self.subimage_tokens*j:self.subimage_tokens*(j+1),:]
                        # print(subimage_selection.shape)
                        subimage_selection = subimage_selection.reshape(-1,self.num_expert)
                        # print(torch.sum(subimage_selection, dim=0))
                        top_subimage_values,top_subimage_index = torch.topk(torch.sum(subimage_selection, dim=0),2)
                        gt_logit = torch.zeros(self.num_expert,device=clean_logits.device)
                        gt_logit[top_subimage_index[0]]=top_subimage_values[0]
                        gt_logit[top_subimage_index[1]]=top_subimage_values[1]
                        # print(top_subimage_values,top_subimage_index,gt_logit)
                        gt_logit = gt_logit.repeat(subimage_selection.shape[0],1)
                        print('gt_logit',gt_logit.shape)
                        # gt_logit = torch.softmax(gt_logit)
                        kl1 = F.kl_div(subimage_selection.softmax(dim=-1).log(), gt_logit.softmax(dim=-1), reduction='batchmean')
                        # kl2 = F.kl_div(gt_logit.softmax(dim=-1).log(), subimage_selection.softmax(dim=-1), reduction='batchmean')
                        self.regu_subimage_loss=self.regu_subimage_loss+kl1 #(kl1+kl2)/2
            self.regu_subimage_loss = self.regu_subimage_loss/(batch_size*30*40/self.subimage_tokens/self.subimage_tokens)



        if self.no_noise:
            noise_stddev *= 0

        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)

        if self.select_idx is not None:
            assert len(self.select_idx) >= self.top_k
            noisy_logits = noisy_logits[:, self.select_idx]

        logits = noisy_logits

        if self.return_decoupled_activation:
            clean_logits_aux = inp @ self.w_gate_aux
            raw_noise_stddev = self.noise_std / self.tot_expert
            noise_stddev_aux = (torch.randn_like(clean_logits) * raw_noise_stddev) * self.training

            if self.no_noise:
                noise_stddev_aux *= 0

            noisy_logits_aux = clean_logits_aux + (torch.randn_like(clean_logits_aux) * noise_stddev_aux)

        if self.select_idx is not None and len(self.select_idx) == self.top_k:
            top_k_gates, top_k_indices = logits.topk(
                min(self.top_k, self.tot_expert), dim=1
            )

            return (
                top_k_indices,
                top_k_gates,
            )

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.tot_expert), dim=1
        )

        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = top_k_logits

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_logits)

        if self.training:
            if self.top_k < self.tot_expert and (not self.no_noise) and abs(noise_stddev) > 1e-6:
                # print("calculate load loss")
                load = (
                    self._prob_in_top_k(
                        clean_logits, noisy_logits, noise_stddev, top_logits
                    )
                ).sum(0)
            else:
                load = self._gates_to_load(gates)

            importance = gates.sum(0)
            loss = self.cv_squared(importance) + self.cv_squared(load)
        else:
            loss = 0

        self.set_loss(loss)
        self.activation = logits.reshape(other_dim + [-1,]).contiguous()

        # print("top_k_indices are {}".format(top_k_indices))
        if self.return_decoupled_activation:
            # print("set activation as noisy_logits_aux")
            self.activation = noisy_logits_aux.reshape(other_dim + [-1, ]).contiguous()

        top_k_indices = top_k_indices.reshape(other_dim + [self.top_k]).contiguous()
        top_k_gates = top_k_gates.reshape(other_dim + [self.top_k]).contiguous()
        # print('top_k_indices',top_k_indices.shape,top_k_gates.shape)
        return (
            top_k_indices,
            top_k_gates,
        )

    def get_activation(self, clear=True):
        activation = self.activation
        if clear:
            self.activation = None
        return activation

    @property
    def has_activation(self):
        return self.activation is not None
