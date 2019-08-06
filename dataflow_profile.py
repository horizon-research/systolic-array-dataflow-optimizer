'''
The functions below are to profile the impacts of different bandwidth,
buffer size, and systolic array size on overall system.
'''
def profile_sa_size(low, high, step):
    for scale in range(1,7):
        arr = []
        # systolic_arr_size, memory_bandwidth, buffer_size
        for size in range(low, high+step, step):
            config = hardware_constraints(sa_size=16.0, mem_bw=32.0, buf=1048576.0*scale*0.5)
            print("SIZE", size)
            config[0] = float(size)
            print(config)
            setup_hardware(config)
            res = opti_dnn()
            arr.append([item[0:4] for item in res])

        # gather results
        res = {"sa_avg":[], "sa_std":[], "buf_avg":[], "buf_std":[], \
                "cycle_avg": [], "traffic_avg": []}

        for ls in arr:
            res["sa_avg"].append(np.mean([i[2] for i in ls]))
            res["sa_std"].append(np.std([i[2] for i in ls]))
            res["buf_avg"].append(np.mean([i[3] for i in ls]))
            res["buf_std"].append(np.std([i[3] for i in ls]))
            res["cycle_avg"].append(int(np.mean([i[1] for i in ls])))
            res["traffic_avg"].append(int(np.mean([i[0] for i in ls])))

        print(res)
        print >> sys.stderr, str(res["traffic_avg"])
        # plot_sa_size(res, low, high, step)
        # plot_sa_cycle(res, low, high, step)

def profile_bw_size(low, high):
    arr = []
    # systolic_arr_size, memory_bandwidth, buffer_size
    config = hardware_constraints()
    for i in range(low, high):
        config[1] = float(pow(2, i))
        setup_hardware(config)
        res = opti_dnn()
        arr.append([item[0:4] for item in res])

    # gather results
    res = {"sa_avg":[], "sa_std":[], "buf_avg":[], "buf_std":[], \
            "cycle_avg": [], "traffic_avg": []}

    for ls in arr:
        res["sa_avg"].append(np.mean([i[2] for i in ls]))
        res["sa_std"].append(np.std([i[2] for i in ls]))
        res["buf_avg"].append(np.mean([i[3] for i in ls]))
        res["buf_std"].append(np.std([i[3] for i in ls]))
        res["cycle_avg"].append(np.mean([i[1] for i in ls]))
        res["traffic_avg"].append(np.mean([i[0] for i in ls]))

    plot_bw_size(res, low, high)
    plot_bw_cycle(res, low, high)

def profile_buf_size(low, high, step, scale):
    arr = []
    # systolic_arr_size, memory_bandwidth, buffer_size
    for size in [0.25, 0.5, 1, 1.5, 2.0, 3.0, 6.0]:
        config = hardware_constraints()
        # config[2] = config[2]*0.125*size
        config[2] = config[2]*size
        print(config)
        setup_hardware(config)
        res = opti_dnn()
        arr.append([item[0:4] for item in res])

    # gather results
    res = {"sa_avg":[], "sa_std":[], "buf_avg":[], "buf_std":[], \
            "cycle_avg": [], "traffic_avg": []}

    for ls in arr:
        res["sa_avg"].append(np.mean([i[2] for i in ls]))
        res["sa_std"].append(np.std([i[2] for i in ls]))
        res["buf_avg"].append(np.mean([i[3] for i in ls]))
        res["buf_std"].append(np.std([i[3] for i in ls]))
        res["cycle_avg"].append(np.mean([i[1] for i in ls]))
        res["traffic_avg"].append(np.mean([i[0] for i in ls]))

    print(res)
    plot_buf_cycle(res, [0.25, 0.5, 1, 1.5, 2.0, 3.0, 6.0], 1)
