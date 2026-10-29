import unittest
from unittest.mock import patch
import numpy as np
from astc_kv.codec import encode_tensor, attention, fp16_floor, fp16_ceil, affine_quantize


class IdentityAstc:
    """Bypass lossy coding to test the independent address/range transform."""
    def __init__(self, block, *args): self.block=block
    def close(self): pass
    def roundtrip(self, im):
        h,w,_=im.shape
        return bytes(((h+self.block-1)//self.block)*((w+self.block-1)//self.block)*16),im.copy(),0.0


class NumericTests(unittest.TestCase):
    def test_transforms_and_permutation_inverses(self):
        x=np.random.default_rng(42).normal(size=(3,288,64)).astype('f4')
        configs=[{'layout':'scalar','block':b,'normalization':'page_channel'} for b in (4,6,8)]
        configs += [{'layout':'rgba','block':4,'normalization':'page_channel'},
                    {'layout':'scalar','block':6,'normalization':'token'},
                    {'layout':'scalar','block':6,'normalization':'block_min','profile':'hdr'}]
        configs += [dict(layout='scalar',block=6,normalization='page_channel',permutation=p) for p in ('tokens_within_page','channels')]
        with patch('astc_kv.codec.AstcEncoder',IdentityAstc):
            for cfg in configs:
                y=encode_tensor(x,cfg)
                np.testing.assert_allclose(y.decoded,x,rtol=2e-6,atol=2e-6,err_msg=str(cfg))

    def test_payload_and_metadata_padding(self):
        with patch('astc_kv.codec.AstcEncoder',IdentityAstc):
            x=np.zeros((3,288,64),np.float32)
            p=encode_tensor(x,dict(layout='scalar',block=6,normalization='page_channel'))
            self.assertEqual(p.width,66)
            self.assertEqual(p.byte_counts['astc_blocks_including_edge_padding'],3*48*11*16)
            self.assertEqual(p.logical_bytes,3*288*64*4//8)
            self.assertEqual(len(p.astc_file(0)),16+48*11*16)
            t=encode_tensor(x,dict(layout='scalar',block=6,normalization='token'))
            self.assertAlmostEqual(t.bits_per_scalar,4+1/6)

    def test_no_future_in_incomplete_page(self):
        with self.assertRaises(ValueError):
            encode_tensor(np.zeros((1,97,64),'f4'),dict(layout='scalar',block=6,normalization='token'))

    def test_metadata_bounds(self):
        x=np.concatenate([np.linspace(-500,500,10001),[0,2**-27,-2**-27,1.0004]]).astype('f4')
        self.assertTrue(np.all(fp16_floor(x).astype('f4')<=x))
        self.assertTrue(np.all(fp16_ceil(x).astype('f4')>=x))

    def test_gqa_causal_against_scalar_oracle(self):
        rng=np.random.default_rng(17)
        q=rng.normal(size=(6,2,8)).astype('f4')
        k=rng.normal(size=(2,7,8)).astype('f4');v=rng.normal(size=k.shape).astype('f4')
        positions=np.array([3,6]);got,p=attention(q,k,v,positions)
        ref=np.zeros_like(got)
        for h in range(6):
            for i,t in enumerate(positions):
                scores=k[h//3,:t+1].astype('f8')@q[h,i].astype('f8')/np.sqrt(8)
                weights=np.exp(scores-scores.max());weights/=weights.sum()
                ref[h,i]=weights@v[h//3,:t+1]
        np.testing.assert_allclose(got,ref,rtol=2e-6,atol=2e-6)
        self.assertTrue(np.all(p[:,0,4:]==0))

    def test_real_astc_constants(self):
        x=np.full((1,96,64),3.75,np.float32)
        for normal,profile in [('page_channel','ldr'),('token','ldr'),('block_min','hdr')]:
            y=encode_tensor(x,dict(layout='scalar',block=6,normalization=normal,profile=profile),quality=10)
            np.testing.assert_allclose(y.decoded,x,atol=1e-5)

    def test_quantizer_counts(self):
        x=np.random.default_rng(4).normal(size=(3,288,64)).astype('f4')
        for bits in (2,4,8):
            decoded,counts=affine_quantize(x,bits)
            self.assertEqual(sum(counts.values())*8/x.size,bits+0.5)
            self.assertTrue(np.isfinite(decoded).all())


if __name__=='__main__':unittest.main()
