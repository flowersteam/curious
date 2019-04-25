# Curious: Intrinsically Motivated Multi-Task Multi-Goal Reinforcement Learning

Implementation of [CURIOUS: Intrinsically Motivated Multi-Task Multi-Goal Reinforcement Learning](https://arxiv.org/abs/1810.06284).

This implementation is based on the [OpenAI baseline](https://github.com/openai/baselines) implementation of [Hindisght Experience Replay](https://arxiv.org/abs/1707.01495) and [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) (included in this repo).

This implementation requires the installation of the [gym_flowers](https://github.com/flowersteam/gym_flowers) module, which overrides gym to enable the use of custom environments such as the one we use in this paper (Multi-Task Multi-Goal Fetch Arm).

The video of the results can be seen [here](https://www.youtube.com/watch?v=qO_OZpsXXGQ&feature=youtu.be)

To run an experiment, run:

python3 /curious/baselines/her/experiment/train.py

options include:
* --num_cpu: Number of cpus. The paper uses 19 cpus (as in the [original paper](https://arxiv.org/abs/1802.09464) presenting this HER implementation. Running the code with fewer cpus for a longer time is NOT equivalent.
* --env: string of the gym_flowers env. Possible choices are MultiTaskFetchArm4-v5 (4 tasks: Reach, Push, Pick and Place, Stack), MultiTaskFetchArm8-v5 (same with 4 distracting tasks).
* --task_selection: use 'active_competence_progress' to use learning progress to guide task selection, 'random' otherwise.
* --goal_selection: 'random' is the only supported here.
* --goal_replay: 'her' uses [Hindisght Experience Replay](https://arxiv.org/abs/1707.01495) or 'none'.
* --task_replay: 'replay_task_cp_buffer' uses learning progress to sample into task-relevant replay buffers. 'replay_task_random_buffer' samples into a buffer associated to a random task.
* --structure: 'curious' uses the curious algorithm, 'task_experts' uses one UVFA policy per task.
* --trial_id: trial identifier.

Results are saved in: /curious/baselines/her/experiment/save/env_name/trial_id/
