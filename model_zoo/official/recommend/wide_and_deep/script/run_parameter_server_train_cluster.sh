#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

execute_path=$(pwd)
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")

#bash run_parameter_server_train_cluster.sh RANK_SIZE EPOCHS DATASET RANK_TABLE_FILE 
#                                            LOCAL_WORKER_NUM LOCAL_SERVER_NUM SERVER_NUM 
#                                            SCHED_HOST SCHED_PORT ROLE
export RANK_SIZE=$1
export EPOCH_SIZE=$2
export DATASET=$3
export RANK_TABLE_FILE=$4

export MS_COMM_TYPE=zmq
export MS_SCHED_NUM=1
export MS_WORKER_NUM=$RANK_SIZE
export LOCAL_WORKER_NUM=$5
export LOCAL_SERVER_NUM=$6
export MS_SERVER_NUM=$7
export MS_SCHED_HOST=$8
export MS_SCHED_PORT=$9
export MS_ROLE=${10}
echo  "=====Role is $MS_ROLE======"


if [ "$MS_ROLE" == "MS_SCHED" ];then
for((i=0;i<1;i++));
do
  rm -rf ${execute_path}/sched_$i/
  mkdir ${execute_path}/sched_$i/
  cd ${execute_path}/sched_$i/ || exit
  export RANK_ID=$i
  export DEVICE_ID=$i
  python -s ${self_path}/../train_and_eval_parameter_server.py --data_path=$DATASET --epochs=$EPOCH_SIZE --parameter_server=1 >sched_$i.log 2>&1 &
done
fi

if [ "$MS_ROLE" == "MS_PSERVER" ];then
for((i=0;i<$LOCAL_SERVER_NUM;i++));
do
  rm -rf ${execute_path}/server_$i/
  mkdir ${execute_path}/server_$i/
  cd ${execute_path}/server_$i/ || exit
  export RANK_ID=$i
  export DEVICE_ID=$i
  python -s ${self_path}/../train_and_eval_parameter_server.py --data_path=$DATASET --epochs=$EPOCH_SIZE --parameter_server=1 >server_$i.log 2>&1 &
done
fi

if [ "$MS_ROLE" == "MS_WORKER" ];then
for((i=0;i<$LOCAL_WORKER_NUM;i++));
do
  rm -rf ${execute_path}/worker_$i/
  mkdir ${execute_path}/worker_$i/
  cd ${execute_path}/worker_$i/ || exit
  export RANK_ID=$i
  export DEVICE_ID=$i
  python -s ${self_path}/../train_and_eval_parameter_server.py --data_path=$DATASET --epochs=$EPOCH_SIZE --parameter_server=1 >worker_$i.log 2>&1 &
done
fi
