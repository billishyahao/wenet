export DNNL_VERBOSE=0
export PYTORCH_TENSOREXPR=0
export PYTHONPATH=/home/yahao/pingan_wenet/wenet:%PYTHONPATH


base_core=0
physical_cores=96

core_arr=(4 2 1)
# core_arr=(4)

log_root="./offline_log/"

echo "saving logs to ${log_root}"
mkdir -p ${log_root}

network="wenet-torchscript"
PRECISION="fp32"

# for j in $(seq 0 $[${#core_arr[@]}-1]); do

#     num_cores=${core_arr[j]}
#     num_groups=$[${physical_cores}/${num_cores}]


#     for k in $(seq 0 $[${num_groups}-1]); do
#         start_core=$[${k}*${num_cores}+${base_core}]
#         end_core=$[$[${k}+1]*${num_cores}-1+${base_core}]
#         benchmark_log="${log_root}/${network}-${PRECISION}-${num_cores}cores_bs${batch_size}_cores${start_core}-${end_core}.log"
#         echo "benchmarking using ${network} with ${num_cores} cores and batchsize=${batch_size} on cores: ${start_core}-${end_core}"
#         echo "saving logs to ${benchmark_log}"; echo
        
#         numactl --physcpubind=${start_core}-${end_core} \
#         python ./wenet/bin/bench_wenet_ipex.py \
# 			--config /home/yahao/pingan_wenet/20210601_u2++_conformer_exp/train.yaml \
# 			--checkpoint /home/yahao/pingan_wenet/20210601_u2++_conformer_exp/final.pt \
# 			--mode torchscript \
# 			--profile true | tee ${benchmark_log} 2>&1 &

#     done
#     wait
#     sleep 10
# done



network="wenet-ipex"
PRECISION="fp32"


# for j in $(seq 0 $[${#core_arr[@]}-1]); do

#     num_cores=${core_arr[j]}
#     num_groups=$[${physical_cores}/${num_cores}]


#     for k in $(seq 0 $[${num_groups}-1]); do
#         start_core=$[${k}*${num_cores}+${base_core}]
#         end_core=$[$[${k}+1]*${num_cores}-1+${base_core}]
#         benchmark_log="${log_root}/${network}-${PRECISION}-${num_cores}cores_bs${batch_size}_cores${start_core}-${end_core}.log"
#         echo "benchmarking using ${network} with ${num_cores} cores and batchsize=${batch_size} on cores: ${start_core}-${end_core}"
#         echo "saving logs to ${benchmark_log}"; echo
        
#         export LD_PRELOAD=/root/.conda/envs/wenet/lib/libiomp5.so:$LD_PRELOAD
#         export LD_PRELOAD=/root/.conda/envs/wenet/lib/libjemalloc.so:$LD_PRELOAD
#         export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
#         export OMP_NUM_THREADS=${num_cores}
#         #export OMP_PROC_BIND=CLOSE
#         #export OMP_SCHEDULE=STATIC
#         export KMP_BLOCK_TIME=1
#         export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
#         #export KMP_AFFINITY=granularity=fine,verbose,compact

#         numactl --physcpubind=${start_core}-${end_core} \
#         python ./wenet/bin/bench_wenet_ipex.py \
# 			--config /home/yahao/pingan_wenet/20210601_u2++_conformer_exp/train.yaml \
# 			--checkpoint /home/yahao/pingan_wenet/20210601_u2++_conformer_exp/final.pt \
# 			--mode ipex \
# 			--profile false | tee ${benchmark_log} 2>&1 &

#     done
#     wait
#     sleep 10
# done


network="wenet-ipex"
PRECISION="bf16"


for j in $(seq 0 $[${#core_arr[@]}-1]); do

    num_cores=${core_arr[j]}
    num_groups=$[${physical_cores}/${num_cores}]


    for k in $(seq 0 $[${num_groups}-1]); do
        start_core=$[${k}*${num_cores}+${base_core}]
        end_core=$[$[${k}+1]*${num_cores}-1+${base_core}]
        benchmark_log="${log_root}/${network}-${PRECISION}-${num_cores}cores_bs${batch_size}_cores${start_core}-${end_core}.log"
        echo "benchmarking using ${network} with ${num_cores} cores and batchsize=${batch_size} on cores: ${start_core}-${end_core}"
        echo "saving logs to ${benchmark_log}"; echo
    	
	export LD_PRELOAD=/root/.conda/envs/wenet/lib/libiomp5.so:$LD_PRELOAD
        export LD_PRELOAD=/root/.conda/envs/wenet/lib/libjemalloc.so:$LD_PRELOAD
        export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
        export OMP_NUM_THREADS=${num_cores}
        #export OMP_PROC_BIND=CLOSE
        #export OMP_SCHEDULE=STATIC
        export KMP_BLOCK_TIME=1
        export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
        #export KMP_AFFINITY=granularity=fine,verbose,compact
        
        numactl --physcpubind=${start_core}-${end_core} \
        python ./wenet/bin/bench_wenet_ipex.py \
			--config /home/yahao/pingan_wenet/20210601_u2++_conformer_exp/train.yaml \
			--checkpoint /home/yahao/pingan_wenet/20210601_u2++_conformer_exp/final.pt \
			--mode ipex_bf16 \
			--profile false | tee ${benchmark_log} 2>&1 &

    done
    wait
    sleep 10
done
