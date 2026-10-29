import itertools
import unittest
import numpy as np
from scripts.analyze_v_error import probabilities, weighted, error_diagnostic, sse


class ErrorDiagnosticTests(unittest.TestCase):
    def test_zero_error_identity(self):
        result,correction=error_diagnostic(np.full((1,1,4),0.25),np.zeros((1,4,2)),2)
        self.assertEqual(result['attention_error_sse'],0)
        self.assertEqual(result['fp16_page_corrected_sse'],0)
        self.assertIsNone(result['original_over_sign_expectation'])
        np.testing.assert_array_equal(correction,0)

    def test_diagonal_expectation_against_exhaustive_signs(self):
        p=np.array([[[0.1,0.2,0.3,0.4]],[[0.4,0.3,0.2,0.1]]])
        error=np.array([[[1.,2.],[-3.,1.],[2.,-2.],[4.,3.]]])
        result,_=error_diagnostic(p,error,2)
        cases=[sse(weighted(p,error*np.array(signs)[None,:,None]))
               for signs in itertools.product([-1.,1.],repeat=4)]
        self.assertAlmostEqual(result['sign_noise_expected_sse'],np.mean(cases),places=12)

    def test_coherent_and_cancelling_errors_same_mse(self):
        p=np.full((1,1,4),0.25)
        coherent=np.ones((1,4,2))
        cancelling=coherent*np.array([1.,-1.,1.,-1.])[None,:,None]
        a,_=error_diagnostic(p,coherent,4)
        b,_=error_diagnostic(p,cancelling,4)
        self.assertEqual(a['v_error_sse'],b['v_error_sse'])
        self.assertGreater(a['cross_token_excess_sse'],0)
        self.assertLess(b['cross_token_excess_sse'],0)
        self.assertEqual(a['fp16_page_corrected_sse'],0)
        self.assertEqual(b['attention_error_sse'],0)

    def test_mask_and_gqa_with_irrelevant_future_key(self):
        q=np.ones((4,1,2))
        k=np.zeros((2,4,2)); k[:,-1]=1000
        p=probabilities(q,k,np.array([1]))
        np.testing.assert_array_equal(p[:,:,2:],0)
        np.testing.assert_allclose(p[:,:,:2],0.5)
        v=np.array([[[2.,3.]]*4,[[5.,7.]]*4])
        np.testing.assert_allclose(weighted(p,v)[:,0],[[2,3],[2,3],[5,7],[5,7]])

    def test_prefix_error_does_not_renormalize_attention(self):
        p=np.array([[[0.1,0.2,0.3,0.4]]])
        v=np.arange(8,dtype='f8').reshape(1,4,2)
        changed=v.copy(); changed[:,:2]+=2
        error=changed[:,:2]-v[:,:2]
        delta=weighted(p,changed)-weighted(p,v)
        np.testing.assert_allclose(delta,weighted(p[:,:,:2],error),rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(delta,0.6)


if __name__=='__main__':
    unittest.main()
