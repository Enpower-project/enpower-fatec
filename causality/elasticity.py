import os
import dill
import numpy as np


class Elasticity:
    def __init__(self, self_el):

        self.self_el = self_el
        self.pem = self.generate_pem()

    def generate_pem(self):

        # Compute all types of PEM defined
        pem_types_ = {
            'Inflexible': self.inflexible(hours=5),
            'EarlyHoursShifting': self.early_hours_shifting(),
            'LateHoursShifting': self.late_hours_shifting(),
            'BackwardShifting': self.backward_shifting(),
            'ForwardShifting': self.forward_shifting(),
            'Flexible': self.flexible(),
            'SelfElastic': self.self_elastic(),
            'RealWorldUsers': self.real_world_users()
        }

        return pem_types_

    def inflexible(self, hours):
        pem_ = np.zeros((len(self.self_el), len(self.self_el)))
        pem_[np.diag_indices_from(pem_)] = self.self_el
        for hour in range(1, hours + 1):
            np.fill_diagonal(pem_[hour:, :-hour], - self.self_el / (hours * 2))
            np.fill_diagonal(pem_[:-hour, hour:], - self.self_el[hour:] / (hours * 2))

        return pem_.T

    def early_hours_shifting(self):
        pem_ = np.zeros((len(self.self_el), len(self.self_el)))
        hour_start, hour_end = 0, 6
        pem_[hour_start:(hour_end + 1), :] = 1
        pem_ = pem_ * (- self.self_el / (hour_end - hour_start + 1))
        pem_[np.diag_indices_from(pem_)] = self.self_el

        return pem_.T

    def late_hours_shifting(self):
        pem_ = np.zeros((len(self.self_el), len(self.self_el)))
        hour_start, hour_end = 17, 23
        pem_[hour_start:(hour_end + 1), :] = 1
        pem_ = pem_ * (- self.self_el / (hour_end - hour_start + 1))
        pem_[np.diag_indices_from(pem_)] = self.self_el

        return pem_.T

    def backward_shifting(self):
        pem_ = self.flexible()
        pem_ = np.tril(pem_)

        return pem_

    def forward_shifting(self):
        pem_ = self.flexible()
        pem_ = np.triu(pem_)

        return pem_

    def flexible(self, beta=0.95, gamma=0.3, alpha=0.05):
        off_start1, off_end1 = 7, 12
        off_start2, off_end2 = 17, 23
        peak_start, peak_end = 13, 16
        valley_start, valley_end = 0, 6

        pem_off_ = np.zeros((len(self.self_el), len(self.self_el)))
        pem_off_[off_start1:(off_end1 + 1), :] = 1
        pem_off_[off_start2:(off_end2 + 1), :] = 1
        pem_off_ *= (- self.self_el * beta * gamma /
                     ((off_end1 - off_start1 + 1) + (off_end2 - off_start2 + 1)))

        pem_peak_ = np.zeros((len(self.self_el), len(self.self_el)))
        pem_peak_[peak_start:(peak_end + 1), :] = 1
        pem_peak_ *= (- self.self_el * alpha / (peak_end - peak_start + 1))

        pem_valley_ = np.zeros((len(self.self_el), len(self.self_el)))
        pem_valley_[valley_start:(valley_end + 1), :] = 1
        pem_valley_ *= (- self.self_el * beta * (1 - gamma) / (valley_end - valley_start + 1))

        pem_ = pem_off_ + pem_peak_ + pem_valley_
        pem_[np.diag_indices_from(pem_)] = self.self_el

        return pem_.T

    def self_elastic(self):
        pem_ = np.zeros((len(self.self_el), len(self.self_el)))
        pem_[np.diag_indices_from(pem_)] = self.self_el

        return pem_.T

    def real_world_users(self):
        pem_ = np.zeros((len(self.self_el), len(self.self_el)))
        for i in range(len(self.self_el)):
            for j in range(len(self.self_el)):
                if j == i:
                    pem_[j, i] = self.self_el[i]
                else:
                    pem_[j, i] = - self.self_el[i] / ((np.abs(i - j) * 1.1)**2 + 1)
        return pem_.T

    @staticmethod
    def save(all_pems):
        os.makedirs(os.path.dirname('pem/'), exist_ok=True)
        for name_, pem_ in all_pems.pem.items():
            with open(f'pem/pem_{name_}.dill', 'wb') as f:
                dill.dump(pem_, f)
