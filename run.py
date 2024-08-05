import os

ip_list = [
    "xxx.xx.x.xx",
    "xxx.xx.x.xx",
    "xxx.xx.x.xx",
    "xxx.xx.x.xx"
]
port = 12345
for ip in ip_list:
    cmd = f"ssh root@{ip}"
    cmd += " '"
    cmd += f"cd {os.getcwd()};"
    cmd += f" PATH={os.environ['PATH']}"
    cmd += " NCCL_SOCKET_IFNAME=eth0"
    cmd += " NCCL_SOCKET_NTHREADS=4"
    cmd += " NCCL_NSOCKS_PERTHREAD=4"
    cmd += " NCCL_ALGO=Ring"
    cmd += " CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
    cmd += " torchrun --nproc_per_node 8"
    cmd += f" --nnodes {len(ip_list)}"
    cmd += f" --node_rank {ip_list.index(ip)}"
    cmd += f" --master_addr {ip_list[0]}"
    cmd += f" --master_port {port}"
    cmd += " train.py"
    cmd += f"  --data_path /xxx/imagenet"
    cmd += f"  --depth=16"
    cmd += f"  --bs=2304"
    cmd += f"  --ep=1000"
    cmd += f"  --fp16=1 --alng=1e-3 --wpe=0.1"
    cmd += "' &"
    print(cmd)
    os.system(cmd)
