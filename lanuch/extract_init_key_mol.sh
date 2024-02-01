cd /remote-home/zhanglu/weakly_molecular

#task_name=tox21_SR-MMP
task_name=tox21_NR-AhR
nohup python -u prepare_data/extract_init_key_mol.py config/$task_name/supervised.py >./lanuch/$task_name/extract_init_key.txt &
