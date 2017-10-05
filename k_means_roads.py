import numpy as np

def k_means(k, data):
    initial_group_indices = np.random.choice(len(data), size=k, replace=False)

    groups = []
    group_means = []

    for i in initial_group_indices:
        group_means.append(data[i])

    group_means = np.array(group_means)

    change_in_means = 1
    run = 0
    while change_in_means > 0:
        print("Run {}".format(run))
        run += 1
        print(change_in_means)

        prev_groups_means = np.copy(group_means)

        groups = []
        for i in initial_group_indices:
            groups.append([])

        groups, group_means = update_group_and_means(data,groups, group_means)
        change_in_means = np.sum(abs(prev_groups_means - group_means))

    return group_means


def update_group_and_means(data, groups, group_means):
    for data_entry in data:
        min_dist = None
        i_closest_group = -1

        for i_group in range(len(group_means)):
            dist = dist_f(data_entry, group_means[i_group])
            if min_dist is None or dist < min_dist:
                min_dist = dist
                i_closest_group = i_group

        groups[i_closest_group].append(data_entry)

    for i_group in range(len(group_means)):
        group_means[i_group] = np.mean(groups[i_group], axis=0)

    return groups, group_means

def dist_f(a,b):
    return np.sum(abs(a - b))


if __name__ == "__main__":
    f = open('roads_matrix.npy','rb')
    matrices = np.load(f)
    f.close()

    group_means = k_means(10, matrices)
