# src/fl/strategy.py
import flwr as fl
from typing import List, Tuple, Dict
from flwr.common import Metrics

class DualFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        # split CT / MRI parameters
        ct_params, mri_params = [], []
        for res in results:
            params = res.parameters.tensors
            n = len(params) // 2
            ct_params.append( (params[:n], res.num_examples // 2) )
            mri_params.append( (params[n:], res.num_examples // 2) )
        agg_ct  = self.aggregate(ct_params)
        agg_mri = self.aggregate(mri_params)
        return agg_ct + agg_mri, {}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict:
    accs = [num * m["accuracy"] for num, m in metrics]
    ns   = [num for num, _ in metrics]
    return {"accuracy": sum(accs)/sum(ns) if ns else 0.0}