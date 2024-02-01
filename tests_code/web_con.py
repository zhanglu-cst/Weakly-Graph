import datetime
import socket

import requests


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.108.255.249', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


Req = requests.Session()
ip = "10.176.26.81"
# r = Req.post('http://10.108.255.249/get_permits.php',data='username=17110240001')
r = Req.post(url = 'http://10.108.255.249/include/auth_action.php?action=login', data = {'action': 'login', \
                                                                                         'username': '21110240014', \
                                                                                         'password': 'ZHANGLU.fudan123', \
                                                                                         'user_ip': ip, \
                                                                                         'ac_id': '1', \
                                                                                         'save_me': '0', \
                                                                                         'ajax': '1'})
