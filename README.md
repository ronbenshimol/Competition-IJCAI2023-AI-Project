
# Competition_IJCAI2023 competition project


## Navigation

This is the folders structures of our project, we included the competiton's code base as our code uses parts of this code, this code should be skipped of course.

### the main folders are:

agents - a folder containing submitted agents
rl_trainer - a folder containing everything needed for the agent trainer we wrote
rl_trainer.agent_trainer.agent_models - a folder containing classes for each rl model supported in the agent trainer
rl_trainer.algo - a folder containing RL algorithm implementations
rl_trainer.rl-algo-competition - Directory we created in the cloud environment for training agents and the game classifier

```

  ┣  env                                    // A competition's folder
  ┣  olympics_engine                        // A competition's folder
  ┣  utils                                  // A competition's folder
  ┣  requirements.txt
  ┣  run_log.py        // Sample file from the competition to train an agent
  ┃
  ┣  agents        // Folder containing submitted agents
  ┃  ┣  dueling_dqn        // First submitted agent, trained using dueling-dqn algorithm
  ┃  ┃  ┣  DuelingDQN.pt        // Pytorch trained dueling-dqn model
  ┃  ┃  ┣  dueling_dqn_agent_submit.py        // Python file with function that return player action
  ┃  ┃  ┗  submission.py        // Sample submission file
  ┃  ┣  game_classifier        // Second submitted agent, utilising game classification
  ┃  ┃  ┣  football_actor.pth        // PPO model to play the football game
  ┃  ┃  ┣  model-game-classifier-loss023-acc-0905.pth // Game classifier model
  ┃  ┃  ┣  running_actor.pth        // PPO model to play running
  ┃  ┃  ┣  submission.py        // Python file to get the player’s next move, it feeds the player’s view to the game classification model, and accordingly then to the specific game’s model
  ┃  ┃  ┗  __init__.py
  ┃  ┗  random        // Random agent submission, used to test the submission requirements.
  ┃  ┃  ┣  random_agent.py
  ┃  ┃  ┗  submission.py
  ┃
  ┗  rl_trainer        // Package containing everything needed for the agent trainer we wrote
  ┃  ┣  agent_trainer
  ┃  ┃  ┣  agent_models        // Package exporting classes for each rl model supported in the agent trainer
  ┃  ┃  ┃  ┣  agent_model_factory.py        // Factory for creating an agent model, for easy configuration purposes 
  ┃  ┃  ┃  ┣  base_agent_model.py
  ┃  ┃  ┃  ┣  ppo_agent_model.py
  ┃  ┃  ┃  ┣  random_agent_model.py
  ┃  ┃  ┃  ┗  __init__.py
  ┃  ┃  ┣  agent_trainer_saves        // Folder for exporting trained rl models
  ┃  ┃  ┃  ┣  ppo_vs_random
  ┃  ┃  ┃  ┃  ┣  2023-06-21_03-49-41
  ┃  ┃  ┃  ┃  ┃  ┗  episode_29600
  ┃  ┃  ┃  ┃  ┃  ┃  ┣  actor.pth
  ┃  ┃  ┃  ┃  ┗  ┗  ┗  critic.pth
  ┃  ┃  ┣  game_environments        // Folder for the different game environments, following base interface
  ┃  ┃  ┃  ┣  base_game_environment.py
  ┃  ┃  ┃  ┣  football_game_environment.py
  ┃  ┃  ┃  ┣  game_environment_factory.py
  ┃  ┃  ┃  ┣  running_game_environment.py
  ┃  ┃  ┃  ┣  table_hockey_game_environment.py
  ┃  ┃  ┃  ┣  wrestling_game_environment.py
  ┃  ┃  ┃  ┗  __init__.py
  ┃  ┃  ┣  agent_trainer.py        // Main agent trainer code (class AgentTrainer)
  ┃  ┃  ┣  argument_parser.py        // Argument parser for running the agent trainer from command line
  ┃  ┃  ┣  dueling_dqn_agent.py        // Script we used to train the first agent submission
  ┃  ┃  ┣  util.py        // Util dataclass for storing transitions
  ┃  ┃  ┗  __init__.py
  ┃  ┣  algo        // RL algorithm implementations
  ┃  ┃  ┣  network.py
  ┃  ┃  ┣  ppo.py
  ┃  ┃  ┗  random.py
  ┃  ┣  models        // Different models we created and trained in AWS for playing the game
  ┃  ┃  ┗  running-competition
  ┃  ┃  ┃  ┗  DuelingDQN
  ┃  ┃  ┃  ┃  ┗  run1
  ┃  ┃  ┃  ┃  ┃  ┗  trained_model
  ┃  ┃  ┃  ┃  ┃  ┃  ┣  qnetwork_stable_5300.pth
  ┃  ┃  ┃  ┃  ┃  ┃  ┣  qnetwork_stable_5300score_-399.0.pth
  ┃  ┃  ┃  ┃  ┃  ┃  ┣  qnetwork_target_5300.pth
  ┃  ┃  ┃  ┃  ┃  ┃  ┗  qnetwork_target_5300score_-399.0.pth
  ┃  ┣  rl-algo-competition        // Directory we created in the cloud environment for training agents and the game classifier
  ┃  ┃  ┣  agents
  ┃  ┃  ┃  ┣  dqn_agent.py
  ┃  ┃  ┃  ┣  duelingdqn_agent.py
  ┃  ┃  ┃  ┗  random_agent.py
  ┃  ┃  ┣  networks
  ┃  ┃  ┃  ┣  dqn_net.py
  ┃  ┃  ┃  ┣  duelingdqn_net.py
  ┃  ┃  ┃  ┗  game_classifier.py



```
---

