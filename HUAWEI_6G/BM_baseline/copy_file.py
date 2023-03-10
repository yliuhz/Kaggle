import moxing as mox
obs_dir='obs://2022-baseline/BM_baseline/model/' #your own dir
mox.file.copy_parallel('./data_test/', obs_dir) 
mox.file.copy_parallel('./models/task_all/model.pth',obs_dir+'model.pth')
mox.file.copy_parallel('./config.json',obs_dir+'config.json')
mox.file.copy_parallel('./customize_service.py',obs_dir+'customize_service.py')
mox.file.copy_parallel('./model.py',obs_dir+'model.py')