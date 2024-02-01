cd /remote-home/zhanglu/weakly_molecular

task_name=sider_Vascular
#task_name=tox21_NR-AhR
nohup python -u pipeline/pipeline2.py config/$task_name/baseline.py >./lanuch/$task_name/baseline.txt &
#nohup python -u pipeline/pipeline2.py config/$task_name/ssl_nodm.py >./lanuch/$task_name/ssl_nodm.txt &
nohup python -u pipeline/pipeline1.py config/$task_name/full.py >./lanuch/$task_name/full.txt &
