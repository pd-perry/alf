import torch
from typing import Union, List, Callable

import alf
from alf.algorithms.td_loss import TDLoss, TDQRLoss
from alf.data_structures import LossInfo, namedtuple, StepType
from alf.utils.losses import element_wise_squared_loss
from alf.utils import losses, tensor_utils, value_ops
from alf.utils.summary_utils import safe_mean_hist_summary
from alf.utils.normalizers import AdaptiveNormalizer

@alf.configurable
class MultiAgentOneStepTDLoss(TDLoss):
    def __init__(self,
                 gamma: Union[float, List[float]] = 0.99,
                 td_error_loss_fn: Callable = losses.element_wise_squared_loss,
                 debug_summaries: bool = False,
                 name: str = "OneStepTDLoss"):
        """
        Args:
            gamma: A discount factor for future rewards. For
                multi-dim reward, this can also be a list of discounts, each
                discount applies to a reward dim.
            td_error_loss_fn: A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            debug_summaries: True if debug summaries should be created
            name: The name of this loss.
        """
        super().__init__(
            gamma=gamma,
            td_error_loss_fn=td_error_loss_fn,
            debug_summaries=debug_summaries,
            td_lambda=0.0,
            name=name)

    def forward(self, info: namedtuple, value: torch.Tensor,
                target_value: torch.Tensor, noise: None):
        """Calculate the loss.

        The first dimension of all the tensors is time dimension and the second
        dimesion is the batch dimension.

        Args:
            info: experience collected from ``unroll()`` or
                a replay buffer. All tensors are time-major. ``info`` should
                contain the following fields:
                - reward:
                - step_type:
                - discount:
            value: the time-major tensor for the value at each time
                step. The loss is between this and the calculated return.
            target_value: the time-major tensor for the value at
                each time step. This is used to calculate return. ``target_value``
                can be same as ``value``.
        Returns:
            LossInfo: with the ``extra`` field same as ``loss``.
        """
        returns = torch.zeros((target_value.shape[0]-1, target_value.shape[1], target_value.shape[2]))
        for n in range(target_value.shape[2]):
            returns_n = self.compute_td_target(info, target_value[:, :, n])
            returns[:, :, n] = returns_n
        
        value = value[:-1]

        if self._normalize_target:
            if self._target_normalizer is None:
                self._target_normalizer = AdaptiveNormalizer(
                    alf.TensorSpec(value.shape[2:]),
                    auto_update=False,
                    debug_summaries=self._debug_summaries,
                    name=self._name + ".target_normalizer")

            self._target_normalizer.update(returns)
            returns = self._target_normalizer.normalize(returns)
            value = self._target_normalizer.normalize(value)

        if self._debug_summaries and alf.summary.should_record_summaries():
            mask = info.step_type[:-1] != StepType.LAST
            with alf.summary.scope(self._name):

                def _summarize(v, r, td, suffix):
                    alf.summary.scalar(
                        "explained_variance_of_return_by_value" + suffix,
                        tensor_utils.explained_variance(v, r, mask))
                    safe_mean_hist_summary('values' + suffix, v, mask)
                    safe_mean_hist_summary('returns' + suffix, r, mask)
                    safe_mean_hist_summary("td_error" + suffix, td, mask)

                if value.ndim == 2:
                    _summarize(value, returns, returns - value, '')
                else:
                    td = returns - value
                    for i in range(value.shape[2]):
                        suffix = '/' + str(i)
                        _summarize(value[..., i], returns[..., i], td[..., i],
                                   suffix)
        
        if noise != None:
            returns = returns + noise[:-1, :, :]
        loss = self._td_error_loss_fn(returns.detach(), value)
        loss = torch.squeeze(loss.sum(dim=-1))

        if loss.ndim == 3:
            # Multidimensional reward. Average over the critic loss for all dimensions
            loss = loss.mean(dim=2)

        # The shape of the loss expected by Algorith.update_with_gradient is
        # [T, B], so we need to augment it with additional zeros.
        loss = tensor_utils.tensor_extend_zero(loss)
        
        return LossInfo(loss=loss, extra=loss)