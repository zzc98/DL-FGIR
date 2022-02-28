# -*- coding: UTF-8 -*-
"""
@Project ：GraduationProject 
@File    ：run_baseline.py
@Author  ：ZhangZichao
@Date    ：2021/3/22 11:03 
"""
from sys import argv

if argv[1] == 'resnet50':
    from baselines import resnet50

    resnet50.main()
elif argv[1] == 'nts':
    from baselines import nts

    nts.main()
elif argv[1] == 'mmal':
    from baselines import mmal

    mmal.main()
elif argv[1] == 'pmg':
    from baselines import pmg

    pmg.main()
elif argv[1] == 'wsdan':
    from baselines import wsdan

    wsdan.main()
else:
    assert False, 'no baseline'
