"""Fast offline regression tests for MLP-only run reliability."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from mmcyber.train import _factorized_specs, _build_model_id
from mmcyber.disagreement import compute_disagreement
from mmcyber.explanation_disagreement import compute_explanation_disagreement
from mmcyber.reevaluate import reconstruct_validation_metrics
from mmcyber.shap_analysis import compute_shap
from mmcyber.aggregate import aggregate_runs
from mmcyber.data import PreparedData


class ReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(Path('configs/default.yaml').read_text(encoding='utf-8'))

    def test_factorized_plan_origin_disjoint_and_fractions(self):
        plans = [_factorized_specs(self.config, origin) for origin in (1, 2)]
        self.assertEqual([len(p) for p in plans], [63, 63])
        for plan in plans:
            self.assertEqual(len({x['model_id'] for x in plan}), 63)
            self.assertEqual({x['subset_fraction'] for x in plan if x['comparison_block'] == 'subset'}, {0.75, 0.5})
            self.assertEqual({x['model_family'] for x in plan}, {'mlp'})
        self.assertTrue({x['train_seed'] for x in plans[0]}.isdisjoint({x['train_seed'] for x in plans[1]}))

    def test_grid_model_ids_are_unique(self):
        values=[]
        for seed in [1, 2]:
            for fraction in [1.0, 0.5]:
                for dims in ['64x32','128x64']:
                    for act in ['relu','tanh']:
                        values.append(_build_model_id(dict(origin_seed=1,comparison_block='grid',model_family='mlp',
                            train_seed=seed,data_seed=seed,subset_fraction=fraction,architecture_id='arch_'+dims,activation=act)))
        self.assertEqual(len(values),len(set(values)))

    def test_validation_rashomon_selection_and_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run';run.mkdir()
            pd.DataFrame([
                dict(model_id='a',seed=1,train_seed=1,data_seed=1,origin_seed=1,comparison_block='baseline',model_family='mlp',
                    subset_strategy='stratified',subset_fraction=1.0,architecture_id='arch_4',hidden_dims_label='4',activation='relu',
                    val_accuracy=0.90,accuracy=0.10,macro_f1=0.1),
                dict(model_id='b',seed=2,train_seed=2,data_seed=1,origin_seed=1,comparison_block='seed',model_family='mlp',
                    subset_strategy='stratified',subset_fraction=1.0,architecture_id='arch_4',hidden_dims_label='4',activation='relu',
                    val_accuracy=0.89,accuracy=0.99,macro_f1=0.9),
            ]).to_csv(run/'metrics.csv',index=False)
            rows=[]
            for model_id,preds in [('a',[0,1,0]),('b',[1,1,0])]:
                for sample_id,ypred in enumerate(preds):
                    rows.append(dict(model_id=model_id,sample_id=sample_id,y_true=sample_id%2,y_pred=ypred))
            pd.DataFrame(rows).to_csv(run/'test_predictions.csv',index=False)
            compute_disagreement(run,rashomon_tolerance=0.02)
            self.assertEqual(set(pd.read_csv(run/'rashomon_models.csv').model_id),{'a','b'})
            summaries=[];shap_values=[]
            for model_id in ('a','b'):
                for feature,value in [('f1',0.2),('f2',-0.1 if model_id=='a' else 0.1)]:
                    summaries.append(dict(model_id=model_id,class_name='attack',feature=feature,rank=1 if feature=='f1' else 2,
                                          mean_abs_shap=abs(value)))
                    for sample_id in [0,1,2]:
                        shap_values.append(dict(model_id=model_id,sample_id=sample_id,class_name='attack',feature=feature,shap_value=value))
            pd.DataFrame(summaries).to_csv(run/'shap_summary.csv',index=False)
            pd.DataFrame(shap_values).to_csv(run/'shap_values_long.csv.gz',index=False,compression='gzip')
            compute_explanation_disagreement(run,top_k=2)
            self.assertTrue((run/'explanation_multiplicity_summary.csv').exists())
            self.assertTrue((run/'shap_variability_summary.csv').exists())
            (run/'config.resolved.json').write_text(json.dumps(dict(dataset=dict(name='test',task='binary'))))
            origin,summary=aggregate_runs([run],Path(tmp)/'aggregated',n_bootstrap=10)
            self.assertEqual(len(origin),1)
            self.assertEqual(len(summary),1)
            self.assertTrue('explanation_mean_abs_shap_cosine_distance_mean' in summary.columns)

    def test_reevaluate_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);(run/'models').mkdir()
            (run/'models'/'a.pt').touch()
            pd.DataFrame([dict(model_id='a',model_path='models/a.pt',accuracy=0.5,val_accuracy=0.4)]).to_csv(run/'metrics.csv',index=False)
            fake=dict(val_accuracy=0.8,val_macro_f1=0.7,val_log_loss=0.5,
                      recomputed_test_accuracy=0.6,recomputed_test_macro_f1=0.5,recomputed_test_log_loss=0.4)
            with patch('mmcyber.reevaluate._load_cached_dataset',return_value={}),\
                 patch('mmcyber.reevaluate._evaluate_mlp',return_value=fake):
                first=reconstruct_validation_metrics(run,device_name='cpu')
                second=reconstruct_validation_metrics(run,device_name='cpu')
            self.assertAlmostEqual(float(first.iloc[0].val_accuracy),0.8)
            self.assertEqual(second.columns.tolist().count('val_accuracy'),1)
            self.assertFalse(any(c.endswith('_x') or c.endswith('_y') for c in second.columns))

    def test_shap_resume_and_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);(run/'models').mkdir()
            (run/'models'/'a.pt').touch()
            cfg=dict(dataset=dict(name='placeholder'),training=dict(device='cpu'))
            (run/'config.resolved.json').write_text(json.dumps(cfg))
            pd.DataFrame([dict(model_id='a')]).to_csv(run/'rashomon_models.csv',index=False)
            pd.DataFrame([dict(sample_id=0,is_conflict=True)]).to_csv(run/'sample_disagreement.csv',index=False)
            data=PreparedData(x_train=np.zeros((5,2),dtype=np.float32),x_val=np.zeros((2,2),dtype=np.float32),
                x_test=np.zeros((3,2),dtype=np.float32),y_train=np.array([0,1,0,1,0]),y_val=np.array([0,1]),
                y_test=np.array([0,1,0]),feature_names=['f1','f2'],class_names=['normal'],preprocessor=None)
            calls=[]
            def worker(payload):
                calls.append(payload)
                ids=payload['explain_idx']
                return dict(model_id='a',sample_indices=ids,shap_values=np.zeros((1,len(ids),2)),
                    summary_rows=[dict(model_id='a',class_name='normal',rank=1,feature='f1',mean_abs_shap=0.0)],
                    value_rows=[dict(model_id='a',sample_id=int(i),class_name='normal',feature='f1',shap_value=0.0) for i in ids])
            with patch('mmcyber.shap_analysis.prepare_dataset',return_value=data),\
                 patch('mmcyber.shap_analysis._compute_model_shap',side_effect=worker):
                compute_shap(run,max_background=2,max_explain=2)
                compute_shap(run,max_background=2,max_explain=2)
                self.assertEqual(len(calls),1)
                compute_shap(run,max_background=3,max_explain=2)
                self.assertEqual(len(calls),2)
            self.assertTrue((run/'shap_values_long.csv.gz').exists())


if __name__=='__main__':
    unittest.main()
