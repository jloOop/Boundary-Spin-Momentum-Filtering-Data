"""Analytic edge cases for finite-window and refinement definitions."""
import unittest
import numpy as np
from centered.numerics import events,linear_integral,linear_abs_difference,step_abs_difference
class IntervalTests(unittest.TestCase):
    def test_straddling_step(self):
        r=events([0,2],[2,4],[.2,.4],1,3,2)
        self.assertAlmostEqual(r["probability"],.15)
        self.assertAlmostEqual(r["first_moment"],.325)
        self.assertAlmostEqual(r["conditional_mean"],13/6)
    def test_partial_linear_survival(self):
        self.assertAlmostEqual(linear_integral([0,2,4],[1,.6,.2],1,3),1.2)
    def test_zero_crossing(self):
        self.assertAlmostEqual(linear_abs_difference([0,1],[1,-1],[0,.3,1],[0,0,0],0,1),.5)
    def test_tiny_zero_crossing(self):
        area=linear_abs_difference([0,1],[1e-170,-1e-170],[0,1],[0,0],0,1)
        self.assertAlmostEqual(area/1e-170,.5)
    def test_native_union_grid(self):
        self.assertAlmostEqual(step_abs_difference([0,1],[1,2],[.1,.3],[0,.5,2],[.5,2,3],[.1,.3,.2],0,2),.2)
    def test_two_exit_accounting(self):
        S,D,E=.2,.3,.5
        self.assertAlmostEqual(S+D+E,1)
        self.assertAlmostEqual(S+E,1-D)
        self.assertNotAlmostEqual(D,1-S)
    def test_outside_support(self):
        with self.assertRaises(ValueError):events([0],[1],[.2],0,2)
    def test_validity_mask_required(self):
        with self.assertRaises(ValueError):linear_integral([0,1],[1,np.nan])
    def test_gap_rejected(self):
        with self.assertRaises(ValueError):events([0,2],[1,3],[.1,.1])
if __name__=="__main__":unittest.main()
