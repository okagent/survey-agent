#!/bin/bash

current_dir=$(pwd)
# 切换到指定目录
cd /home/yxf/WIP/sva/scipdf_parser-master

# # 在后台启动服务
nohup bash serve_grobid.sh &
GROBID_PID=$(pgrep -f 'appname=gradlew')

sleep 30

cd "$current_dir"
python arxiv_update_daily.py

if [ ! -z "$GROBID_PID" ]; then
    echo "Stopping GROBID service with PID: $GROBID_PID"
    kill -SIGTERM "$GROBID_PID"
    # 等待进程结束
    wait "$GROBID_PID" 2>/dev/null
    echo "GROBID service stopped."
else
    echo "GROBID PID not found."
fi