import sys

def count_digit(n):
    count = 0
    while n > 0:
        n = n // 10
        count += 1
    return count


def read_train_data(data_file='9Cat-Train.labeled', encoding='utf-8'):
    with open(data_file ,'r' ,encoding = encoding) as f:
        lines = f.readlines()
        data_list = []
    for index ,i in enumerate(lines):
        i = i.split('\t')
        list_in_list = []

        for j in i:
            j = j.split(' ')[1]
            list_in_list.append(j.strip('\n'))
        data_list.append(list_in_list)

    attr_list = []
    label_list = []

    for i in data_list:
        attr_list.append(i[:-1])
        label_list.append(i[-1])
    return attr_list ,label_list

def Find_s(attr_list ,label_list ,fname = 'partA6.txt'):
    attr_len = len(attr_list[0])

    specific_h = ['null' ] *attr_len

    with open(fname ,'w') as f:
        for index1 ,i in enumerate(attr_list):
            if label_list[index1] == 'Yes':
                for index2 ,a in enumerate(specific_h):
                    if (a == 'null'):
                        specific_h[index2] = i[index2]
                    elif (a != i[index2]):
                        specific_h[index2] = '?'
                    else:
                        specific_h[index2] = a
            if ((index1 +1) % 20 == 0):
                for index,i in enumerate(specific_h):
                    if (index +1) == len(specific_h):
                        f.write('%s\n '% (i))
                    else:
                        f.write('%s\t '% (i))

    return specific_h


def read_dev_data(data_file='9Cat-Dev.labeled', encoding='utf-8'):
    with open(data_file, 'r', encoding=encoding) as f:
        lines = f.readlines()
        data_list = []
    for index, i in enumerate(lines):
        i = i.split('\t')
        list_in_list = []

        for j in i:
            j = j.split(' ')[1]
            list_in_list.append(j.strip('\n'))
        data_list.append(list_in_list)

    dev_attr_list = []
    dev_label_list = []
    for i in data_list:
        dev_attr_list.append(i[:-1])
        dev_label_list.append(i[-1])
    return dev_attr_list, dev_label_list


def cal_classification(dev_attr_list, dev_label_list, specific_h):
    null_index = []
    any_index = []
    spe_index = []

    for index, i in enumerate(specific_h):
        if i == 'null':
            null_index = list(range(7))
        elif i == '?':
            any_index.append(index)
        else:
            spe_index.append(index)

    dev_pre_label_list = []

    # null situations
    if specific_h == ['null'] * 7:
        for i in dev_attr_list:
            dev_pre_label_list.append('No')

    # all any situations
    elif specific_h == ['?'] * 7:
        for i in dev_attr_list:
            dev_pre_label_list.append('Yes')

    # other situations
    else:
        for i in dev_attr_list:
            if [i[index] for index in spe_index] == [specific_h[index] for index in spe_index]:
                dev_pre_label_list.append('Yes')
            else:
                dev_pre_label_list.append('No')

    count = 0
    for index, i in enumerate(dev_pre_label_list):
        if i == dev_label_list[index]:
            count += 1

    rate = count / len(dev_label_list)
    mis_rate = 1.00 - rate

    return mis_rate

def read_std_data(data_file = sys.argv[1], encoding='utf-8'):
    with open(data_file, 'r', encoding=encoding) as f:
        lines = f.readlines()
        data_list = []
    for index, i in enumerate(lines):
        i = i.split('\t')
        list_in_list = []

        for j in i:
            j = j.split(' ')[1]
            list_in_list.append(j.strip('\n'))
        data_list.append(list_in_list)

    test_attr_list = []
    test_label_list = []
    for i in data_list:
        test_attr_list.append(i[:-1])
        test_label_list.append(i[-1])

    return test_attr_list, test_label_list


def cal_test_classification(test_attr_list, test_label_list, specific_h):
    null_index = []
    any_index = []
    spe_index = []

    for index, i in enumerate(specific_h):
        if i == 'null':
            null_index = list(range(7))
        elif i == '?':
            any_index.append(index)
        else:
            spe_index.append(index)

    test_pre_label_list = []

    # null situations
    if specific_h == ['null'] * 7:
        for i in test_attr_list:
            test_pre_label_list.append('No')

    # all any situations
    elif specific_h == ['?'] * 7:
        for i in test_attr_list:
            test_pre_label_list.append('Yes')

    # other situations
    else:
        for i in test_attr_list:
            if [i[index] for index in spe_index] == [specific_h[index] for index in spe_index]:
                test_pre_label_list.append('Yes')
            else:
                test_pre_label_list.append('No')

    # count = 0
    # for index, i in enumerate(dev_pre_label_list):
    #     if i == dev_label_list[index]:
    #         count += 1
    #
    # rate = count / len(dev_label_list)
    # return rate
    for i in test_pre_label_list:
        print(i)

def main():
    # Input space
    Input_space = 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2
    print(Input_space)
    # Concept space
    con_space = 2 ** (Input_space)

    print(count_digit(con_space))
    # hypothesis space
    hypothesis_space = 1 + (3 * 3 * 3 * 3 * 3 * 3 * 3 * 3 * 3)
    print(hypothesis_space)
    # new hypothesis space
    new_hypothesis_space = 1 + (3 * 3 * 3 * 3 * 3 * 3 * 3 * 3 * 3 * 3)
    print(new_hypothesis_space)
    # new hypothesis space 2
    new_hypothesis_space_2 = 1 + (4 * 3 * 3 * 3 * 3 * 3 * 3 * 3 * 3)
    print(new_hypothesis_space_2)
    attr_list, label_list = read_train_data()
    dev_attr_list, dev_label_list = read_dev_data()
    specific_h = Find_s(attr_list, label_list)
    dev_rate = cal_classification(dev_attr_list, dev_label_list, specific_h)
    print(dev_rate)
    test_attr_list, test_label_list = read_std_data()
    cal_test_classification(test_attr_list, test_label_list, specific_h)

if __name__ == "__main__":
    main()