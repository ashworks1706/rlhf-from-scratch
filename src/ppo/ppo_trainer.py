import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from transformers import Trainer
from core_utils import masked_mean, masked_var, masked_whiten


class PPOTrainer(Trainer):
    def __init__(
        self,
        args = None,
        ppo_engine = None,
        data_collator = None,
        train_dataset = None,
        tokenizer = None,
        optimizer = Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = [None, None]
        **kwargs
    ):
        self.args = args
        self.actor_model = ppo_engine.actor_model
        self.critic_model = ppo_engine.critic_model

        self.model = PPOModel(self.actor_model, self.critic_model)

    def get_params(self, model, lr, weight_decay, eps=1e-8):
        params = [
            {
                "params": [p for n, p in model.named_parameters() if model.requires_grad()],
                "weight_decay": weight_decay,
                "lr": lr,
                "eps": eps
            }
        ]
        return params

    def create_optimizer(self):
        params = self.get_params(self.actor_model, self.args.actor_lr, self.args.actor_weight_decay)
        params.extend(self.get_params(self.critic_model, self.args.critic_lr, self.args.critic_weight_decay))

        optimizer = torch.optim.Optimizers.AdamW(params, betas = (0.9, 0.95))
        return optimizer
    
    def create_scheduler(self, optimizer, max_update_steps):
        params = self.get_params(self.actor_model, self.args.actor_lr, self.args.actor_weight_decay)
        params.extend(self.get_params(self.crttic_model, self.args.critic_lr, self.args.critic_weight_decay))

    def get_last_reward_score(self, values, responses_mask):
        """
        Computes the critic's last-state value estimates.

        Args:
            values: torch.Tensor
                the critic model's outputs for all tokens (baseline V(s_t))
            responses_mask: torch.Tensor
                binary mask marking which tokens belong to the generated response
        """
        batch_size = values.shape[0] 
        reward_score = []
        for i in range(batch_size):
            value = values[i] 
            end_idx = responses_mask[i].nonzero()[-1].detach().item() # last token in the sequence
            reward_score.append(value[end_idx]) # collect value prediction at the final token
        rewards_score = torch.stack(reward_score) # collect all last-state value estimates

        return rewards_score
    
    def get_log_probs(self, logits, labels):
        """
        Computes the log-probabilities given logits, and computes the label
        """
        log_probs = F.log_softmax(logits, dim=-1)
        labels = labels.unsqueeze(-1)
        log_probs_labels = log_probs.gather(dim=-1, index=labels)
        return log_probs_labels.squeeze(-1)
    
    def get_entropy(self, logits, mask):
        """
        Computes the entropy of the policy, which incentivizes exploration.

        Args:
            logits: torch.Tensor
                unnormalized log-probabilities from a model
            mask: torch.Tensor
                mask to be applied to the computed entropy
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = self.masked_mean(-torch.sum(probs * log_probs, dim=-1), mask)
        return entropy



    def compute_rewards_with_kl_penalty(self, ref_values, actor_log_probs, ref_log_probs, responses_mask):
        """
        Computes rewards with the KL divergence penalty.
        Includes implementations of KL estimators since we can't compute it exactly.
        - k_3 is a popular KL estimator proposed here: http://joschu.net/blog/kl-approx.html

        Args:
            ref_values

            actor_log_probs: torch.Tensor
                log probabilities from our actor model

            ref_log_probs: torch.Tensor
                log probabilities from our reference model

            responses_mask: torch.Tensor
        """
        masks = responses_mask[:, 1:] 
        rewards_score = self.get_last_reward_score(ref_values, responses_mask)
        
        batch_size = rewards_score.shape[0]
        rewards_with_kl_penalty, kl_penalty_all = [], []

        for i in range(batch_size):
            mask = masks[i]
            lp_a = actor_log_probs[i][mask] # masked actor logprobs
            lp_r = ref_log_probs[i][mask] # masked reference logprobs


            # in my equations below, r is simply the ratio: pi(y) / pi_ref(y)
            if self.args_kl_penalty_method == 'k_3': # equation: (r - 1) - log r
                lp_diff = lp_a - lp_r
                ratio = torch.exp(lp_diff)
                kl_est = (ratio - 1.0) - lp_diff

            elif self.args.kl_penalty_method == 'abs': # equation: |log r|
                kl_est = torch.abs(lp_a - lp_r)

            elif self.args.kl_penalty_method == 'mse': # equation: 1/2 * (log r)^2
                kl_est = 0.5 * (lp_a - lp_r) ** 2 
                
            kl_penalty = - self.args.kl_penalty_beta * kl_est
            kl_penalty_all.append(kl_penalty)

            if self.args.reward_score_clip is not None:
                rewards_score[i] = torch.clamp(rewards_score[i], -self.args.reward_score_clip, self.args.reward_score_clip)
            
            end_index = mask.nonzero()[-1].detach().item()
            kl_penalty[end_index] += rewards_score[i]

            rewards_with_kl_penalty.append(kl_penalty)
        return torch.stack(rewards_with_kl_penalty), torch.stack(kl_penalty_all), rewards_score 

    def compute_gae_advantage_return(self, rewards, values, mask, gamma, lam):
        """
        Computes the Generalized Advantage Estimation via Temporal-Difference with parameter lambda.

        Args:
            rewards: torch.Tensor
                shape: (bs, response_length)
            values: torch.Tensor
                shape: (bs, response_length)
            mask: torch.Tensor
                shape: (bs, response_length)
            gamma: float
                discount factor
            lam: float
                lambda parameter for GAE algorithm
        """
        B, T = rewards.shape # B is batch size, T is response_length

        # here, we bootstrap the value model updates with Temporal-Difference (parameter \lambda)
        with torch.no_grad():
            advantages_reversed = []
            lastgaelam = 0

            for t in reversed(range(T)):
                # for long sequences with T - t >> 1, discounting reduces the reward signal to near zero
                next_values = values[:, t + 1] if t < T - 1 else 0.0
                delta = rewards[:, t] + gamma * next_values - values[:, t]
                lastgaelam = (delta + gamma * lam * lastgaelam) * mask[:, t] 
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], dim=1)

            returns = advantages + values
            # 
            if use_advantage_norm:
                advantages = masked_whiten(advantages, eos_mask)
        return advantages, returns

    def get_responses_mask(self, sequences_mask, prompts_without_padding):
        """
        Computes a mask tensor for responses in a given batch of batch size B.
        """
        B, T = sequences_mask.shape
        responses_mask = []

        for i in range(B):
            prompt = prompts_without_padding[i]
            response_mask = torch.zeros_like(sequences_mask[i])
            response_mask[len(prompt):] = sequences_mask[i][len(prompt):]
            responses_mask.append(response_mask)
        return torch.stack(responses_mask)

    def compute_policy_loss(self, old_log_prob, log_prob, advantages, eos_mask, epsilon):
        """
        Computes the policy gradient loss function for PPO.

        Args:
            old_log_prob: torch.Tensor
                log probabilities from the old policy
            log_prob: torch.Tensor
                log probabilities from the current policy 
            advantages: torch.Tensor
                Computed advantages via advantage estimation
            eos_mask: torch.Tensor
            
            epsilon: float
        """

        # from log domain -> real domain
        ratio = torch.exp(log_prob - old_log_prob)
        loss1 = -advantages * ratio
        loss2 = -advantages * torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

        loss = masked_mean(torch.max(loss1, loss2), eos_mask)


    def compute_value_loss(self, value_preds, returns, values, eos_mask, epsilon):
        """
        Fits the value function by regression on the MSE loss. 

        Args:
            value_preds: torch.Tensor
                the predictions from our value model
            returns: torch.Tensor
                shape: (bs, response_length)
            values: torch.Tensor
                
            eos_mask: torch.Tensor

            epsilon: torch.Tensor
        """
        
        # this keeps our value predictions within some epsilon-distance
        # mostly for numerical stability before performing regression
        clip_value_preds = torch.clamp(value_preds, values - epsilon, values + epsilon)   

        # thus, we have two errors, but it doesn't matter because we take the maximum (one)
        values_error = (value_preds - returns) ** 2
        clip_values_error = (clip_value_preds - returns) ** 2
        
        # this is essentially the inner sum for one trajectory t \in D_k where D_k is set of trajectories
        loss = 0.5 * masked_mean(torch.max(values_error, clip_values_error), eos_mask)
        return loss, values_error
    
    @torch.no_grad()
    def get_model_output(self, sequences):
        """
        Fetches the model outputs from both the actor and the critic 
        """
        actor_output = self.actor_model(**sequences, return_dict=True) # of shape (B, T, V)
        actor_logits = actor_output.logits

        critic_values = self.critic_model(**sequences)[-1] # get the last scalar, of shape (B, T)
        
        # now, fetch reference logits for the KL divergence (from a frozen model)
        ref_output = self.ref_model(**sequences, return_dict=True)

    def get_state_dict(self, model):
        """
        Collects the state dict of our pretrained model. 
        The value head is a linear projection of our pretrained model's last hidden states onto a scalar
        which estimates the value of a state (for variance reduction, advantage computation).
        """
        pretrained_model_state_dict = model.pretrained_model.state_dict()
        value_head_state_dict = model.value_head.state_dict()
        for k, v in value_head_state_dict.items():
            pretrained_model_state_dict[f"value_head.{k}"] = v
        return pretrained_model_state_dict
    
