import openxlab
openxlab.login(ak='r1n6pvon4ej1je6dxwzq', sk='r0n1xo2pyd9xavkvqpjrkdyr574ng5ydorble6mj') # 进行登录，输入对应的AK/SK，可在个人中心添加AK/SK
from openxlab.dataset import get
get(dataset_repo='OpenDataLab/CelebA-HQ', target_path='./data') # 数据集下载

