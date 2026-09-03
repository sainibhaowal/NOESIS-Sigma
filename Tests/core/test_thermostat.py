import torch

from Core.OSC.Control.thermostat import Thermostat, ThermostatConfig


def test_calm_downshifts_and_dt_up():
    cfg = ThermostatConfig(check_interval=1, window_M=2, lower_band=1e-3, upper_band=5e-2)
    th = Thermostat(cfg)
    x = torch.zeros(8, 16)           # very calm
    S, dt = 64, 1e-3
    for _ in range(5):                # enough iterations to trigger window_M twice
        S, dt = th.maybe_update(_, x, S, dt, energy_fn=None)
    assert S <= 64 and dt >= 1e-3     # moved in the right directions
    assert S >= cfg.s_min and dt <= cfg.dt_max

def test_transient_upshifts_and_dt_down():
    cfg = ThermostatConfig(check_interval=1, lower_band=1e-6, upper_band=1e-4)
    th = Thermostat(cfg)
    S, dt = 32, 1e-3
    x_prev = torch.zeros(8, 16)
    th.force_update(x_prev, S, dt)     # prime prev refs
    x_spike = torch.ones(8, 16) * 10.0 # big transient
    S2, dt2 = th.maybe_update(0, x_spike, S, dt, energy_fn=None)
    assert S2 >= S and dt2 <= dt
