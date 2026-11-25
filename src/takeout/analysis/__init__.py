"""Analyses of model output: functions of datasets, independent of the code that ran.

- :mod:`takeout.analysis.instabilities`: what grows in a run, how fast and what it is,
  from an instability record (needs ``takeout[analysis]``);
- :mod:`takeout.analysis.exb`: the dominant azimuthal wave of the E x B benchmarks;
- :mod:`takeout.analysis.rotating_frame`: positions in a frame rotating about ``z``;
- :mod:`takeout.analysis.ion_acoustic`: the kinetic dispersion relation of the
  ion-acoustic wave and the frequency and damping of a recorded mode.

The submodules are imported one by one, so that the optional dependencies of one do not
weigh on the others.
"""
