import numpy as np
from typing import Optional

from .base import BaseHRBMBoosting
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class RegularizedHRBMBoosting(BaseHRBMBoosting):
    def __init__(self, n_estimators: int = 1000,
                 loss: str = 'mse',
                 classification: bool = False,
                 learning_rate: float = 1.0,
                 reg_type: str = 'value',
                 l1_reg: float = 0.0,
                 l2_reg: float = 0.0,
                 l1_reg_min: float = 0.0,
                 l2_reg_min: float = 0.0,
                 reg_beta: float = 1.0,
                 enable_validation: bool = True,
                 val_size: float = 0.5,
                 val_gamma: Optional[float] = None,
                 recalculate_values: bool = True,
                 hold_validation_set: bool = True,
                 n_score_rebuild_attempts: int = 0,
                 n_val_rebuild_tries: int = 0,
                 need_rsm: bool = False,
                 rsm_size: int = -1,
                 margin: float = 0.0,
                 random_state: Optional[int] = None):
        """Initialize Regularized HRBM Boosting.

        Args:
            n_estimators: (Maximum) Number of HRBMs.
            loss: Loss function ('mse', 'ce' or any `LossFn` derivative).
            classification: If True, build classifier, else regressor.
            learning_rate: Learning rate (gamma). Default is 1.0.
            reg_type: Regularization type ('value' or 'height').
            l1_reg: L1 regularization parameter (float or 'auto').
            l2_reg: L2 regularization parameter (float or 'auto').
            l1_reg_min: Minimum L1 regularization parameter for `l1_reg='auto'`.
            l2_reg_min: Minimum L2 regularization parameter for `l2_reg='auto'`.
            reg_beta: Regularization beta, HRBM absolute value upper bound for
                      `l1_reg='auto'` or `l2_reg='auto'`.
            enable_validation: Enable validation.
            val_size: Validation set size.
            val_gamma: Validation learning rate.
            recalculate_values: Recalculate HRBM in, out values using
                                the whole data set after validation.
            hold_validation_set: Use the same validation set for each iteration.
                                 Otherwise resplit data set into train-val each iteration.
            n_score_rebuild_attempts: Number of HRBM construction iterations.
            n_val_rebuild_tries: Number of rebuild tries for HRBMs that
                                 do not pass validation.
            need_rsm: Use Random Subspace Method in HRBMs or not.
            rsm_size: Random Subspace size.
            margin: HRBM feature boundaries margin size.
            random_state: Random state (int, `np.RandomState` instance or None).
        """
        self.n_estimators = n_estimators
        self.loss = loss
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.val_gamma = val_gamma
        self.reg_type = reg_type
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.l1_reg_min = l1_reg_min
        self.l2_reg_min = l2_reg_min
        self.reg_beta = reg_beta
        self.recalculate_values = recalculate_values
        self.hold_validation_set = hold_validation_set
        self.enable_validation = enable_validation
        self.n_score_rebuild_attempts = n_score_rebuild_attempts
        self.n_val_rebuild_tries = n_val_rebuild_tries
        # other parameters
        self.need_rsm = need_rsm
        self.rsm_size = rsm_size
        # common parameters
        self.classification = classification
        self.random_state = random_state
        self.margin = margin

    def __l1_min_feasible(self, n, g_sum, denom):
        d = np.abs(denom)
        case_1_left = -(g_sum + self.reg_beta * d)
        case_1_right = (-g_sum + self.reg_beta * d)
        case_2_left = (g_sum - self.reg_beta * d)
        case_2_right = (g_sum + self.reg_beta * d)
        if (case_1_left <= 0 <= case_1_right) or (case_2_left <= 0 <= case_2_right):
            return 0.0
        opts = [
            np.abs(g_sum),
            case_1_left,
            case_1_right,
            case_2_left,
            case_2_right,
        ]
        return np.min([a for a in opts if a >= 0]) / n

    def __get_l1_reg(self, n, g_in_sum, denom_in, g_out_sum, denom_out):
        if self.l1_reg == 'auto':
            # automatic l1 determination (based on beta)
            return max(
                self.l1_reg_min,
                self.__l1_min_feasible(n, g_in_sum, denom_in),
                self.__l1_min_feasible(n, g_out_sum, denom_out)
            )
        return self.l1_reg

    def __calc_regularized_value(self, n: int,
                                 g_in_sum: float,
                                 h_in_sum: float,
                                 g_out_sum: float,
                                 h_out_sum: float):
        """Calculate regularized value for APR VALUE regularization type.
        """
        # find denominator
        if isinstance(self.l2_reg, float):
            l2_reg = self.l2_reg
        elif self.l2_reg == 'auto':
            # automatic l2 determination (based on beta)
            l2_reg_in = (np.abs(g_in_sum) / self.reg_beta - h_in_sum) / n
            l2_reg_out = (np.abs(g_out_sum) / self.reg_beta - h_out_sum) / n
            l2_reg = max(self.l2_reg_min, np.max(l2_reg_in), np.max(l2_reg_out))
        denom = (n * l2_reg + h_in_sum)
        # find numerator
        if self.l1_reg == 0:
            num = -g_in_sum
        else:
            l1_reg = self.__get_l1_reg(n, g_in_sum, denom, g_out_sum, (n * l2_reg + h_out_sum))
            n_dot_l1 = n * l1_reg
            if g_in_sum < -n_dot_l1:
                num = -(g_in_sum + n_dot_l1)
            elif g_in_sum <= n_dot_l1:
                num = 0
            else:
                num = -(g_in_sum - n_dot_l1)
        return num, denom

    def __get_height_l2_reg(self, n: int,
                            g_in_sum: float,
                            h_in_sum: float,
                            g_out_sum: float,
                            h_out_sum: float):
        if self.l2_reg == 'auto':
            g = g_in_sum + g_out_sum
            h = h_in_sum + h_out_sum
            beta = self.reg_beta
            if g <= -beta * h:
                denom_inv = 1 / (n * (g - beta * h))
                return max(
                    self.l2_reg_min,
                    h_out_sum * (beta * h_in_sum - g_in_sum) * denom_inv,
                    h_in_sum * (beta * h_out_sum - g_out_sum) * denom_inv,
                )
            elif g < beta * h:
                denom_le_inv = 1 / (n * (g - beta * h))
                denom_ge_inv = 1 / (n * (g + beta * h))
                return max(
                    self.l2_reg_min,
                    h_out_sum * (beta * h_in_sum - g_in_sum) * denom_le_inv,
                    h_in_sum * (beta * h_out_sum - g_out_sum) * denom_le_inv,
                    -h_out_sum * (beta * h_in_sum + g_in_sum) * denom_ge_inv,
                    -h_in_sum * (beta * h_out_sum + g_out_sum) * denom_ge_inv,
                )
            else:  # g >= beta * h
                denom_inv = 1 / (n * (g + beta * h))
                return max(
                    self.l2_reg_min,
                    -h_out_sum * (beta * h_in_sum + g_in_sum) * denom_inv,
                    -h_in_sum * (beta * h_out_sum + g_out_sum) * denom_inv,
                )
        return self.l2_reg

    def __calc_regularized_height(self, n: int,
                                 g_in_sum: float,
                                 h_in_sum: float,
                                 g_out_sum: float,
                                 h_out_sum: float):
        """Calculate regularized value for APR HEIGHT regularization type.
        """
        g_sum = g_in_sum + g_out_sum
        h_sum = h_in_sum + h_out_sum
        # step height regularization
        if self.l1_reg == 0:
            # l2 regularization
            l2_reg = self.__get_height_l2_reg(n, g_in_sum, h_in_sum, g_out_sum, h_out_sum)
            num = -(g_in_sum + n * l2_reg * (g_sum / h_out_sum))
            denom = (h_in_sum + n * l2_reg * (h_sum / h_out_sum))
        elif self.l2_reg == 0:
            # l1 regularization
            lhs = (h_in_sum * g_out_sum - h_out_sum * g_in_sum) / h_sum
            n_dot_l1 = n * self.l1_reg
            if lhs < -n_dot_l1:
                num = -(g_in_sum - n_dot_l1)
            elif g_in_sum <= n_dot_l1:
                num = -g_in_sum
            else:
                num = -(g_in_sum + n_dot_l1)
            denom = h_in_sum
        else:
            raise ValueError(
                f'Step height regularization assumes that l1 or l2 == 0'
            )
        return num, denom

    def _calc_value(self, n_in: int,
                    n_out: int,
                    g_in_sum: float,
                    h_in_sum: float,
                    g_out_sum: float,
                    h_out_sum: float):
        n = n_in + n_out
        if self.reg_type == 'value':
            num, denom = self.__calc_regularized_value(n, g_in_sum, h_in_sum, g_out_sum, h_out_sum)
        elif self.reg_type == 'height':
            num, denom = self.__calc_regularized_height(n, g_in_sum, h_in_sum, g_out_sum, h_out_sum)
        else:
            raise ValueError(f'Wrong reg_type: {self.reg_type}')
        return num / denom

    def _fill(self, X, y, init_value, sample_weight=None, rng=None, loss_fn=None):
        train_ind, val_ind = train_test_split(
            np.arange(X.shape[0]),
            test_size=self.val_size,
            random_state=rng
        )

        if not self.enable_validation:
            train_ind = np.arange(X.shape[0])
            val_ind = val_ind[:0]

        gamma = self.learning_rate
        if self.val_gamma is None:
            gamma_val = gamma
        else:
            gamma_val = self.val_gamma

        val_error = np.inf
        to_use = np.full((self.n_estimators,), True, dtype=bool)

        if loss_fn.CONSTANT_HESSIAN:
            h_all = np.full_like(y, 2)
        cum_pred_all = init_value
        mask_in_all = self.hrbms_.mask_in(X)

        i = 0
        n_tries = 0
        while i < self.n_estimators:  # for i in range(self.n_estimators):
            to_use[i] = True  # give the rectangle a chance

            # calculate gradients, hessians
            loss_fn.clear()
            _loss_value = loss_fn(y, cum_pred_all, no_return=True)  # evaluate gradients, hessians
            g_all = loss_fn.gradient
            if not loss_fn.CONSTANT_HESSIAN:
                h_all = loss_fn.hessian
            # otherwise h_all remains constant

            # fill the current rectangle
            use_cur_apr = True
            # fill the current APR using the train data
            n_points_train = train_ind.shape[0]

            # if self.n_score_rebuild_attempts == 0:
            if self.n_score_rebuild_attempts > 0:
                score_attempt = 0
                attempts_costs = []
                attempts_params = []
            mask_in_all_i = mask_in_all[:, i]
            while True:
                mask_in_train_i = mask_in_all_i[train_ind]
                n_in = mask_in_train_i.sum()
                use_cur_apr = True
                if n_in == 0 or n_in == n_points_train:
                    use_cur_apr = False
                    v_in = 0.0
                    v_out = 0.0
                else:
                    g_in_sum = g_all[train_ind][mask_in_train_i].sum(axis=0)
                    g_out_sum = g_all[train_ind][~mask_in_train_i].sum(axis=0)
                    h_in_sum = h_all[train_ind][mask_in_train_i].sum(axis=0)
                    h_out_sum = h_all[train_ind][~mask_in_train_i].sum(axis=0)
                    n_out = n_points_train - n_in
                    v_in = self._calc_value(n_in, n_out, g_in_sum, h_in_sum, g_out_sum, h_out_sum)
                    v_out = self._calc_value(n_out, n_in, g_out_sum, h_out_sum, g_in_sum, h_in_sum)
                if self.n_score_rebuild_attempts == 0:
                    break
                # score the attempt
                # mask_in_val_i = mask_in_all_i[val_ind]
                cur_cum_pred = cum_pred_all[train_ind] + v_out * gamma
                cur_cum_pred[mask_in_train_i] += (v_in - v_out) * gamma
                cur_cost = loss_fn(y[train_ind], cur_cum_pred)
                attempts_costs.append(cur_cost)
                attempts_params.append(
                    (
                        self.hrbms_.lefts[i].copy(),
                        self.hrbms_.rights[i].copy(),
                        v_in,
                        v_out,
                        mask_in_all_i.copy()
                    )
                )
                if score_attempt >= self.n_score_rebuild_attempts:
                    # select the best and stop
                    best_attempt = np.argmin(attempts_costs)
                    l, r, v_in, v_out, mask_in_all_i = attempts_params[best_attempt]
                    self.hrbms_.lefts[i], self.hrbms_.rights[i] = l, r
                    mask_in_all[:, i] = mask_in_all_i
                    # mask_in_train_i = mask_in_all_i[train_ind]
                    break
                else:
                    score_attempt += 1
                    self._rebuild_apr(i, X, rng=rng)
                    mask_in_all_i = self.hrbms_.mask_in_apr(X, i)

            n_in = mask_in_train_i.sum()
            if n_in == 0 or n_in == n_points_train:
                use_cur_apr = False
                v_in = 0.0
                v_out = 0.0

            if self.enable_validation and use_cur_apr:  # APR is not discarded yet
                if self.hold_validation_set:
                    # calculate validation error
                    mask_in_val_i = mask_in_all[val_ind, i]
                    cur_cum_pred_val = cum_pred_all[val_ind] + v_out * gamma_val
                    cur_cum_pred_val[mask_in_val_i] += (v_in - v_out) * gamma_val
                    cur_val_error = loss_fn(y[val_ind], cur_cum_pred_val)
                    if cur_val_error >= val_error:
                        use_cur_apr = False
                else:  # not self.hold_validation_set
                    mask_in_val_i = mask_in_all[val_ind, i]
                    # estimate error on the validation set without update
                    prev_cum_pred_val = cum_pred_all[val_ind]
                    prev_val_error = loss_fn(y[val_ind], prev_cum_pred_val)
                    # ... and with update
                    cur_cum_pred_val = cum_pred_all[val_ind] + v_out * gamma_val
                    cur_cum_pred_val[mask_in_val_i] += (v_in - v_out) * gamma_val
                    cur_val_error = loss_fn(y[val_ind], cur_cum_pred_val)
                    if cur_val_error >= prev_val_error:
                        use_cur_apr = False

            if use_cur_apr:  # APR is not discarded
                mask_in_all_i = mask_in_all[:, i]
                if self.recalculate_values:
                    # recalculate values
                    n_in = mask_in_all_i.sum()
                    n_out = g_all.shape[0] - n_in
                    g_in_sum = g_all[mask_in_all_i].sum(axis=0)
                    g_out_sum = g_all[~mask_in_all_i].sum(axis=0)
                    h_in_sum = h_all[mask_in_all_i].sum(axis=0)
                    h_out_sum = h_all[~mask_in_all_i].sum(axis=0)
                    v_in = self._calc_value(n_in, n_out, g_in_sum, h_in_sum, g_out_sum, h_out_sum)
                    v_out = self._calc_value(n_out, n_in, g_out_sum, h_out_sum, g_in_sum, h_in_sum)

                # update APR values
                self.hrbms_.out_bias += v_out * gamma
                self.hrbms_.in_values[i] = (v_in - v_out) * gamma
                # upadte cumulative prediction & recalculate target for the whole set
                cum_pred_all += v_out * gamma
                cum_pred_all[mask_in_all_i] += (v_in - v_out) * gamma
                # target_all = (cum_pred_all - y)

                # update validation error
                if self.enable_validation:
                    val_error = cur_val_error
                else:
                    val_error = None
            else:  # not use_cur_apr
                if n_tries < self.n_val_rebuild_tries:
                    # rebuild the rectangle
                    self._rebuild_apr(i, X, rng=rng)
                    mask_in_all[:, i] = self.hrbms_.mask_in_apr(X, i)
                    n_tries += 1
                    continue
                else:
                    n_tries = 0
                to_use[i] = False

            if self.enable_validation:
                # train_ind, val_ind = val_ind, train_ind
                if not self.hold_validation_set:
                    train_ind, val_ind = train_test_split(
                        np.arange(X.shape[0]),
                        test_size=self.val_size,
                        random_state=rng
                    )

            i += 1
            # end of while

        self._prune_hrbms(to_use)
        return

    def _prune_hrbms(self, to_use: np.ndarray):
        self.hrbms_.lefts = self.hrbms_.lefts[to_use]
        self.hrbms_.rights = self.hrbms_.rights[to_use]
        self.hrbms_.in_values = self.hrbms_.in_values[to_use]
        self.hrbms_.n_estimators = self.hrbms_.in_values.shape[0]
        self.n_estimators = self.hrbms_.in_values.shape[0]

