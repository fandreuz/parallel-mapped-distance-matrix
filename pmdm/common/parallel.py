def distribute_and_start_subproblems(
    trigger,
    nup_count_per_bin,
    pts_per_future,
    executor,
):
    curr_start = 0
    futures = []
    for bin_idx, count in enumerate(nup_count_per_bin):
        while count > pts_per_future:
            count -= pts_per_future
            futures.append(trigger(bin_idx, curr_start, pts_per_future))
            curr_start += pts_per_future
        if count > 0:
            futures.append(trigger(bin_idx, curr_start, count))
            curr_start += count
    return futures
