
### **Launching Fault-Tolerant and Elastic Training Jobs with `torchrun`**

#### **1. Fault-Tolerant Job Setup**
A fault-tolerant job can restart automatically when nodes or processes fail during training. To launch a fault-tolerant job, use the following command **on all nodes** in your cluster:

```bash
torchrun \
    --nnodes=NUM_NODES \
    --nproc-per-node=TRAINERS_PER_NODE \
    --max-restarts=NUM_ALLOWED_FAILURES \
    --rdzv-id=JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint=HOST_NODE_ADDR \
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

- **NUM_NODES**: Total number of nodes participating in the training.
- **TRAINERS_PER_NODE**: Number of training processes to launch per node.
- **NUM_ALLOWED_FAILURES**: Maximum number of times the job can restart before giving up.
- **JOB_ID**: A unique identifier for this job, used to track nodes in the training job.
- **HOST_NODE_ADDR**: Address of one of the nodes hosting the rendezvous backend (format: `<host>:<port>`, defaults to port `29400` if no port is provided).

---

#### **2. Elastic Job Setup**
Elastic training jobs allow nodes to dynamically join or leave during training. To set it up, run the following command on at least `MIN_SIZE` nodes and at most `MAX_SIZE` nodes:

```bash
torchrun \
    --nnodes=MIN_SIZE:MAX_SIZE \
    --nproc-per-node=TRAINERS_PER_NODE \
    --max-restarts=NUM_ALLOWED_FAILURES_OR_MEMBERSHIP_CHANGES \
    --rdzv-id=JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint=HOST_NODE_ADDR \
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

- **MIN_SIZE**: Minimum number of nodes required to begin training.
- **MAX_SIZE**: Maximum number of nodes that can participate.
- **NUM_ALLOWED_FAILURES_OR_MEMBERSHIP_CHANGES**: Number of allowed failures or changes in cluster membership.
  
---

### **Key Notes**
1. **How Elastic Jobs Handle Failures:**
   - Failures (like a node crashing) are treated as membership changes. A failed node is treated as "scaling down," while replacing it is "scaling up."
   - Use `--max-restarts` to limit the total number of restarts due to failures or membership changes.

2. **`HOST_NODE_ADDR`:**
   - Represents a host and port for the C10d rendezvous backend (e.g., `node1.example.com:29400`).
   - Pick a node with high bandwidth for hosting.

3. **Single-Node Jobs with `--standalone`:**
   - If you’re running on a single node, add the `--standalone` option. You **do not need** to specify `--rdzv-id`, `--rdzv-endpoint`, or `--rdzv-backend`.

---

### **Example Commands**
- Fault-Tolerant Job (2 nodes, 4 processes per node, 3 failures allowed):
  ```bash
  torchrun --nnodes=2 \
           --nproc-per-node=4 \
           --max-restarts=3 \
           --rdzv-id=job123 \
           --rdzv-backend=c10d \
           --rdzv-endpoint=node1.example.com:29400 \
           train.py --arg1 value1 --arg2 value2
  ```

- Elastic Job (minimum 2 nodes, maximum 4 nodes):
  ```bash
  torchrun --nnodes=2:4 \
           --nproc-per-node=4 \
           --max-restarts=5 \
           --rdzv-id=job_elastic \
           --rdzv-backend=c10d \
           --rdzv-endpoint=node1.example.com:29400 \
           train.py --arg1 value1
  ```

---

### **More Customization**
If `torchrun` does not meet your needs, check out the [TorchElastic Agent API](https://pytorch.org/docs/stable/distributed.elastic.html) for advanced use cases. 



