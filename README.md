# final_assignment___Robot_position_control_using_reinforcement_learning

はじめに：  
　容量の関係上、IsaacSim IsaacLabすべては本GitHub内には投稿しておりません。  
  下記手順に従い、環境構築をしてください。  


1.IsaacSim IsaacLabをインストールする。  
　下記URLを基に環境を構築する。  
	https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#isaaclab-installation-root  

2.下記ファイルを本GitHub内の該当ファイルと入れ替える。  

	2-1. C:\IsaacLab\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\__init__ .py  
	2-2. C:\IsaacLab\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\manipulation\reach\mdp\rewards.py  
	2-3. C:\IsaacLab\IsaacLab\source\isaaclab\config\extension.toml  
	
	それぞれ、下記を行っている。  
	　・カスタムタスクを佐パッケージとして明示的にロードする設定を追加  
	　・報酬関数を追加  
	　・カスタムタスクの拡張機能追加  
	
3.下記階層に本GitHub内の該当フォルダを入れる。  

	3-1. C:\IsaacLab\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\manipulation\franka_pose_reach_task  
	
4.学習  
	PS C:\IsaacLab\IsaacLab> C:\IsaacLab\env_IsaacLab\Scripts\activate.ps1  
	(env_IsaacLab) PS C:\IsaacLab\IsaacLab> .\isaaclab.bat -p .\scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Franka-Pose-Reach-v0  
	
	
5.自律運転  
	PS C:\IsaacLab\IsaacLab> C:\IsaacLab\env_IsaacLab\Scripts\activate.ps1  
	(env_IsaacLab) PS C:\IsaacLab\IsaacLab> .\isaaclab.bat -p .\scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Franka-Pose-Reach-Play-v0 --checkpoint <checkpointのパス> --num_envs 1  
	
	checkpointは下記フォルダに格納されている。  
	C:/IsaacLab/IsaacLab/logs/rsl_rl/franka_pose_reach/  
	
	
