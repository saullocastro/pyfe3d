r"""Cost of the constitutive matrices in the element coordinate system

Times :meth:`pyfe3d.shellprop.ShellProp.get_constitutive_element`, called by
all shell elements, in a compiled loop without Python overhead, for laminates
with a different number of plies, see ``CHANGELOG.md``. The repository root
must be importable, e.g.::

    PYTHONPATH=<path to pyfe3d repository> python bench_constitutive_element.py

"""
import os
import sys

import pyximport

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
pyximport.install(language_level=3)

from bench_transverse_shear_cost import pin_process, CFRP
from pyfe3d.shellprop_utils import laminated_plate
import micro_constitutive_element as micro


STACKS = {
    '1 ply': [0],
    '8 plies': [0, 45, -45, 90, 90, -45, 45, 0],
    '32 plies': [0, 45, -45, 90]*8,
    '128 plies': [0, 45, -45, 90]*32,
}


def main():
    pin_process()
    for stack_name, stack in STACKS.items():
        prop = laminated_plate(stack=stack, plyt=0.125e-3, laminaprop=CFRP)
        for full in [0, 1]:
            for thetadeg in [0., 30.]:
                t = min(micro.bench(prop, 200000, thetadeg, full)
                        for _ in range(7))
                print('%-9s %-15s theta=%4.0f  %6.1f ns/call'
                      % (stack_name, 'A, B, D and Ats' if full else 'Ats only',
                         thetadeg, t))


if __name__ == '__main__':
    main()
