1) CUDA_ERROR_OUT_OF_MEMORY probably indicates that you already had one TensorFlow program running and tried to start a second.  One solution is to run each process on different GPU cards  Another option is to tell TF to only use a fraction of the memory.

2) Pinning an operation on a device means telling TF to place a certain operation on a certain device.  If there is no kernel for the operation (no GPU kernal for integer variables), then TF places the operation on a CPU.  This is called soft placement.  Pinning is what you want TF to do.  Placement is what TF ends up doing.

3) By default, TF will run all operations which have a GPU kernel on the first GPU.  If soft placemet is allowed and there is no GPU kernel, the operation will run on the CPU.

4) Variables that are pinned on a certain device can be used by another device.  In disributed sessions, variable state is managed by resource containers located on the cluster itself.

5) 2 ops placed on the same device can run in parallel by using different threads.  TF manages variable dependencies so operations do not get executed before the dependencies do.

5) A control dependency is is the postponement of evaluation of certain nodes, even when the dependencies have been executed.  If an operation uses a lot of RAM but is not needed until the end of the graph, it may make sense to delay it.  Another case is operations dependent on data outside the device.  If they run at the same time, the communication bandwith may get saturated.

6) In distributed Tensorflow the variables live in containers in the clusters, so one only needs to reopen the session and save them.