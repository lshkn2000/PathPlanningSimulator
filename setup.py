from setuptools import setup

setup(name='PathPlanningSimulator',
      version='0.0.1',
      packages=[
          'path_planning_simulator',
          'path_planning_simulator.sim',
          'path_planning_simulator.policy',
      ],
      install_requires=['gym',
                        'torch',
                        'matplotlib',
                        'numpy',
                        'scipy',
                        'tensorboard',
                        'tqdm',
                        'sklearn',
                        'pytorch_lightning',
                        'cma',
                        'opencv-python'])
