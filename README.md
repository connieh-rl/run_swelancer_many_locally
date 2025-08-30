# run_swelancer_many_locally

To run many swe_lancer scenarios locally: 

1. Git clone (https://github.com/openai/preparedness):
2. After cloning, cd into the preparedness folder into the file path: cd preparedness/project/swelancer, and then copy this file run_swe_lancer_locally.py into that folder.
3.  Run the setup commands for swelancer (specified here) (https://github.com/openai/preparedness/tree/main/project/swelancer)
4. Run this script from the root of the repository using Python 3.12+:

python project/swelancer/run_swe_lancer.py --model <MODEL_NAME> [--task_types <TYPE>] [--individual_scenarios <ID1,ID2,...>]
