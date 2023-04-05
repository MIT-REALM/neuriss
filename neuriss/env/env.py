import torch
import os

from .microgrid import Microgrid
from .platoon import Platoon
from .planar_drone import PlanarDrone


def make_env(
        env_id: str,
        device: torch.device,
        dt: float = 0.01,
):
    if env_id == 'Microgrid':
        return Microgrid(device, dt)
    elif env_id == 'Microgrid-IEEE4':
        dir_name = os.path.dirname(os.path.abspath(__file__))
        params = torch.load(dir_name + f'/data/{env_id}.pkl', map_location=device)
        return Microgrid(device, dt, params)
    elif env_id == 'Microgrid-IEEE5':
        dir_name = os.path.dirname(os.path.abspath(__file__))
        params = torch.load(dir_name + f'/data/{env_id}.pkl', map_location=device)
        return Microgrid(device, dt, params)
    elif 'PlatoonSin' in env_id:
        n_systems = env_id.split('PlatoonSin')[1]
        try:
            n_systems = int(n_systems)
        except ValueError:
            raise ValueError('Invalid env id: number of trucks should be an integer.')
        params = {
            'n': n_systems,
            'm': torch.ones(n_systems, device=device),
            'v_init': 2.0,
            'r': 1.0,
            'speed profile': 'sin'
        }
        return Platoon(device, dt, params)
    elif 'Platoon' in env_id:
        n_systems = env_id.split('Platoon')[1]
        try:
            n_systems = int(n_systems)
        except ValueError:
            raise ValueError('Invalid env id: number of trucks should be an integer.')
        params = {
            'n': n_systems,
            'm': torch.ones(n_systems, device=device),
            'v_init': 2.0,
            'r': 1.0,
            'speed profile': 'constant'
        }
        return Platoon(device, dt, params)
    elif 'PlanarDroneConst' in env_id:
        args = env_id.split('PlanarDroneConst')[1]
        n_row = args.split('x')[0]
        n_col = args.split('x')[1]
        try:
            n_row = int(n_row)
            n_col = int(n_col)
        except ValueError:
            raise ValueError('Invalid env id: number of drones should be an integer.')
        params = {
            'n_col': n_col,
            'n_row': n_row,
            'm': torch.ones(n_col * n_row, device=device),
            'I': torch.ones(n_col * n_row, device=device),
            'l': torch.ones(n_col * n_row, device=device) * 0.3,
            'r': 1.,
            'vx_init': 1.,
            'vy_init': 0.,
            'speed profile': 'constant'
        }
        return PlanarDrone(device, dt=0.03, params=params)
    elif 'PlanarDroneSin' in env_id:
        args = env_id.split('PlanarDroneSin')[1]
        n_row = args.split('x')[0]
        n_col = args.split('x')[1]
        try:
            n_row = int(n_row)
            n_col = int(n_col)
        except ValueError:
            raise ValueError('Invalid env id: number of drones should be an integer.')
        params = {
            'n_col': n_col,
            'n_row': n_row,
            'm': torch.ones(n_col * n_row, device=device),
            'I': torch.ones(n_col * n_row, device=device),
            'l': torch.ones(n_col * n_row, device=device) * 0.3,
            'r': 1.,
            'vx_init': 1.,
            'vy_init': 0.,
            'speed profile': 'sin'
        }
        return PlanarDrone(device, dt=0.03, params=params)
    elif 'PlanarDrone' in env_id:
        args = env_id.split('PlanarDrone')[1]
        n_row = args.split('x')[0]
        n_col = args.split('x')[1]
        try:
            n_row = int(n_row)
            n_col = int(n_col)
        except ValueError:
            raise ValueError('Invalid env id: number of drones should be an integer.')
        params = {
            'n_col': n_col,
            'n_row': n_row,
            'm': torch.ones(n_col * n_row, device=device),
            'I': torch.ones(n_col * n_row, device=device),
            'l': torch.ones(n_col * n_row, device=device) * 0.3,
            'r': 1.,
            'vx_init': 0.,
            'vy_init': 0.,
            'speed profile': 'static'
        }
        return PlanarDrone(device, dt=0.03, params=params)
    else:
        raise NotImplementedError(f'{env_id} not implemented')
