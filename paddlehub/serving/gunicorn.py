#!/usr/bin/env python
# coding=utf-8
# coding: utf8
"""
configuration for gunicorn
"""
import multiprocessing
bind = '0.0.0.0:8888'
backlog = 2048
workers = multiprocessing.cpu_count() * 2 + 1
threads = 1
worker_class = 'sync'
worker_connections = 1000
timeout = 500
keepalive = 40
daemon = False
loglevel = 'info'
errorlog = '-'
accesslog = '-'
