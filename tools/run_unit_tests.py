#!/usr/bin/env python3
"""Run numerical/protocol tests; opt in to the separately built native codec."""
import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--with-codec', action='store_true', help='Require the actual native ASTC roundtrip')
    args = ap.parse_args()
    if not args.with_codec:
        from test_numeric import NumericTests
        NumericTests.test_real_astc_constants = unittest.skip(
            'Native ASTC bridge not requested; build it and pass --with-codec'
        )(NumericTests.test_real_astc_constants)
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
