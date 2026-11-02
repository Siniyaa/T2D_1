import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from evaluation import offset_correct, calibration, net_benefit, rescaled_auc_ci

class EvaluationTests(unittest.TestCase):
    def test_each_development_prior_maps_to_target_prior(self):
        for prior in [6547/11429, 3896/9487]:
            self.assertAlmostEqual(offset_correct([prior], prior, .08)[0], .08, places=12)
        with self.assertRaises(ValueError):
            offset_correct([.5], 0, .1)

    def test_offset_preserves_ranking_and_calibration_slope(self):
        # Fixed non-separated observations; offset must move only calibration intercept.
        y = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        p = np.array([.1, .15, .2, .3, .4, .5, .6, .7, .8, .9])
        q = offset_correct(p, .57, .08)
        a, s, b = calibration(y, p)
        c, t, d = calibration(y, q)
        shift = logit(.08)-logit(.57)
        self.assertAlmostEqual(c, a-shift, places=8)
        self.assertAlmostEqual(s, t, places=6)
        self.assertEqual(roc_auc_score(y, p), roc_auc_score(y, q))
        self.assertAlmostEqual(b, np.mean((y-p)**2))

    def test_calibration_weights_match_replicated_observations(self):
        y = np.array([0, 1, 0, 1, 0, 1]); p = np.array([.1,.2,.3,.4,.7,.9])
        w = np.array([1, 3, 2, 4, 2, 1])
        np.testing.assert_allclose(calibration(y,p,w),
            calibration(np.repeat(y,w),np.repeat(p,w)),rtol=1e-6,atol=1e-8)

    def test_weighted_net_benefit_known_value(self):
        self.assertAlmostEqual(net_benefit([1,0,1],[.8,.7,.1],.5,[2,1,1]),.25)
        self.assertEqual(net_benefit([1,0],[0,0],.5),0)

    def test_survey_domain_retains_empty_psu_in_design(self):
        # Two full-design PSUs in each stratum. A domain can omit one of them.
        rows=[]
        for stratum in range(4):
            for psu in [1,2]:
                for y in [0,1]:
                    rows.append(dict(SDMVSTRA=stratum,SDMVPSU=psu,y=y,p=.2+.5*y+.02*psu,w=1.))
        full=pd.DataFrame(rows)
        domain=full[~((full.SDMVSTRA==0)&(full.SDMVPSU==2))]
        point,lo,hi,n,cases=rescaled_auc_ci(domain,full,'p','w',B=100)
        self.assertEqual((point,lo,hi),(1.,1.,1.))
        self.assertEqual(n,len(domain))
        with self.assertRaises(ValueError):
            rescaled_auc_ci(domain,domain,'p','w',B=100)

    def test_weighted_auc_matches_known_ties(self):
        rows=[]
        for stratum in range(3):
            for psu in [1,2]:
                rows.extend([dict(SDMVSTRA=stratum,SDMVPSU=psu,y=1,p=.5,w=2.),
                             dict(SDMVSTRA=stratum,SDMVPSU=psu,y=0,p=.5,w=1.),
                             dict(SDMVSTRA=stratum,SDMVPSU=psu,y=0,p=.1,w=3.)])
        d=pd.DataFrame(rows)
        point,lo,hi,_,_=rescaled_auc_ci(d,d,'p','w',B=100)
        self.assertAlmostEqual(point,.875)
        self.assertAlmostEqual(lo,.875)
        self.assertAlmostEqual(hi,.875)

if __name__ == '__main__': unittest.main()
