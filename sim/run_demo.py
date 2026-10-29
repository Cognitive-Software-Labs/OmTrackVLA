"""Short closed-loop HSSD demo using installed assets; not benchmark evaluation."""
import argparse
import copy
import gzip
import json
import math
import os
import sys
import shutil
from pathlib import Path

import torch
import habitat_sim
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--prepared', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.prepared:
        run_evaluation(args.steps)
        return
    torch.set_num_threads(4)
    root = Path('data/local_demo')
    root.mkdir(parents=True, exist_ok=True)
    source = Path('data/scene_datasets/hssd-hab/stages/102343992.glb')
    scene = root / '102343992.glb'
    if not scene.exists():
        shutil.copyfile(source, scene)
    settings = habitat_sim.SimulatorConfiguration()
    settings.scene_id = str(scene)
    settings.enable_physics = True
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = []
    with habitat_sim.Simulator(habitat_sim.Configuration(settings, [agent])) as sim:
        nav = habitat_sim.NavMeshSettings()
        nav.set_defaults()
        nav.agent_radius = 0.3
        if not sim.recompute_navmesh(sim.pathfinder, nav):
            raise RuntimeError('Could not build demo navigation mesh.')
        sim.pathfinder.seed(100)
        island = max(range(sim.pathfinder.num_islands), key=sim.pathfinder.island_area)
        for _ in range(1000):
            human = sim.pathfinder.get_random_navigable_point(island_index=island)
            robot = sim.pathfinder.get_random_navigable_point_near(human, 2.5, island_index=island)
            distance = np.linalg.norm(robot-human)
            if np.isfinite(robot).all() and 1.5 < distance < 2.5:
                break
        else:
            raise RuntimeError('Could not sample demo starting positions.')
        goal = sim.pathfinder.get_random_navigable_point_near(human, 4, island_index=island)
        navdir = root / 'navmeshes'
        navdir.mkdir(exist_ok=True)
        sim.pathfinder.save_nav_mesh(str(navdir / '102343992.navmesh'))
    with gzip.open('data/datasets/track/STT/val/val.json.gz', 'rt') as f:
        episode = copy.deepcopy(json.load(f)['episodes'][0])
    # The demo samples new positions in an available scene; benchmark episodes stay intact.
    episode.update(episode_id='demo', scene_id=str(scene), scene_dataset_config='default',
                   start_position=human.tolist(), start_rotation=[0, 0, 0, 1],
                   goals=[{'position': [goal.tolist()], 'radius': None}])
    episode['info'] = {
        'robot_position': robot.tolist(),
        # Spot's forward axis is +X; positive yaw turns it toward -Z.
        'robot_rotation': math.atan2(float(robot[2]-human[2]), float(human[0]-robot[0])),
        'human_num': 0, 'num_goals_for_main_human': 1, 'num_goals_for_other_human': 1,
        'main_humanoid_name': 'female_2', 'main_human_semantic_id': 1002,
        'extra_humanoid_names': [], 'episode_mode': 'stt',
        'instruction': 'Follow the target person without collision.',
    }
    dataset_path = root / 'demo.json.gz'
    with gzip.open(dataset_path, 'wt') as f:
        json.dump({'episodes': [episode]}, f)
    # A fresh process avoids retaining a closed EGL context from preparation.
    os.execv(sys.executable, [sys.executable, '-u', '-m', 'sim.run_demo', '--prepared', '--steps', str(args.steps)])


def run_evaluation(steps):
    torch.set_num_threads(4)
    dataset_path = Path('data/local_demo/demo.json.gz')
    import habitat
    import sim.evt_bench  # noqa: F401  (registers Habitat plugins)
    from habitat.config import read_write
    from habitat.datasets import make_dataset
    from habitat.core.registry import registry
    from habitat.tasks.track.track_task import MultiHumanTrackingTask

    @registry.register_task(name='LocalHSSDDemoTrackingTask-v0')
    class LocalHSSDDemoTrackingTask(MultiHumanTrackingTask):
        def _switch_avatar(self, sim, episode):
            # The configured default avatar is already female_2. Retain it:
            # removing it can leave native rendering references to its nodes.
            if episode.info['main_humanoid_name'] != 'female_2':
                raise ValueError('This demo expects the configured female_2 avatar.')
            human = sim.agents_mgr[0].articulated_agent
            for node in human.sim_obj.visual_scene_nodes:
                node.semantic_id = episode.info['main_human_semantic_id']
    from sim.trained_agent import evaluate_agent
    config = habitat.get_config('habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml')
    with read_write(config):
        config.habitat.dataset.data_path = str(dataset_path)
        config.habitat.dataset.scenes_dir = '.'
        config.habitat.environment.max_episode_steps = steps
        config.habitat.task.type = 'LocalHSSDDemoTrackingTask-v0'
        # Only the robot cameras are needed. Humanoid cameras can retain a
        # removed scene node when the tracking task switches avatars.
        for name, agent in config.habitat.simulator.agents.items():
            if name != 'agent_1':
                agent.sim_sensors.clear()
    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    print('Running a local HSSD demo, not the MP3D/HM3D benchmark.', flush=True)
    evaluate_agent(config, dataset, 'outputs/simulator-demo', apply_scene_lighting=False)
    print('Demo complete: outputs/simulator-demo', flush=True)


if __name__ == '__main__':
    main()
