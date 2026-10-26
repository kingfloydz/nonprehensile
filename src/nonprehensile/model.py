"""Shared MuJoCo contact model for offline settling and batched training."""

XML = """
<mujoco model="accelerating_tray">
  <option timestep="0.0005" gravity="0 0 0" cone="elliptic"/>
  <default><geom condim="3" solref="0.004 1" solimp="0.95 0.99 0.001"/></default>
  <worldbody>
    <geom name="tray" type="box" pos="0 0 -0.01" size="0.4 0.4 0.01"
          friction="0.8 0 0" rgba="0.25 0.4 0.55 1"/>
    <body name="box" pos="0 0 0.1">
      <freejoint/>
      <geom name="payload" type="box" size="0.1 0.1 0.1" density="500"
            priority="1" friction="0.5 0 0" rgba="0.85 0.5 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""
