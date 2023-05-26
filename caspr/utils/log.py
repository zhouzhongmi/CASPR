import os
import logging
from datetime import datetime


env_dic = os.environ
jhub_user = env_dic.get('JUPYTERHUB_USER')


class XLogger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info',
                 fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        log_dir = f"/home/{jhub_user}/shared/MI_ZHOU/transformer/CASPR/logs"
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_filename = os.path.join(log_dir, '.'.join([filename, datetime.now().strftime('%Y-%m-%d')]))
        self.logger = logging.getLogger(log_filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        # self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setLevel(self.level_relations.get(level))
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = logging.FileHandler(filename=log_filename, encoding='utf-8')
        th.setLevel(logging.DEBUG)
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)
        self.logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    log = XLogger('caspr.log', level='debug')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('警告')
    log.logger.error('报错')
    log.logger.critical('严重')
    XLogger('error.log', level='error').logger.error('error')
