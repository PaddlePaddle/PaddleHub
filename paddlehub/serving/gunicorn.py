#!/usr/bin/env python
# coding=utf-8
# coding: utf8
"""
configuration for gunicorn
"""
import multiprocessing
bind = '0.0.0.0:8888'
backlog = 2048
workers = multiprocessing.cpu_count()
threads = 2
worker_class = 'sync'
worker_connections = 1000
timeout = 30
keepalive = 2
daemon = False
loglevel = 'info'
errorlog = '-'
accesslog = '-'
