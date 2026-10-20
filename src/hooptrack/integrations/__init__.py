"""Adapters between the two chapters of the project.

The tracker (chapter 1) and the court geometry (chapter 2, ``hooptrack.court``)
never import each other (``tests/court/test_separation.py``). When one needs the
other, the bridge lives here, behind a config switch, so that turning the switch
off gives back exactly the previous behaviour.
"""
