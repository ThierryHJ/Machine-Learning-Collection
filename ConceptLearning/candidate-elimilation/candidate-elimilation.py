import sys
import itertools

def read_train_data(data_file='4Cat-Train.labeled', encoding='utf-8'):
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

    first_occur = []
    new_attr_list = []
    for index,i in enumerate(attr_list):
        for j in range(len(attr_list[0])):
            tem_list = []
            if index == 0:
                first_occur.append(i[j])
                i[j] = 0
            else:
                if i[j] == first_occur[j]:
                    i[j] = 0
                elif i[j] != first_occur[j]:
                    i[j] = 1
            tem_list.append(i[j])
        new_attr_list.append(tem_list)

    return attr_list ,label_list,first_occur

def list_eliminate(attr_list ,label_list):
    attr_len = len(attr_list[0])

    input_space_list = [[0,1] for i in range(attr_len)]
    # print("input_space_list: ",input_space_list)

    input_space = list(itertools.product(*input_space_list))
    # print("input_space:",input_space)
    concept_space_1 = list(itertools.product(*input_space_list,['Yes','No']))
    # print("concept_space_pre: ", concept_space_1)
    # print(len(concept_space_1))
    concept_space_1_list = []
    tem_list = []
    for index,i in enumerate(concept_space_1):
        if index % 2 == 0:
            tem_list.append(i)
        else:
            tem_list.append(i)
            concept_space_1_list.append(tem_list)
            tem_list = []
    # print("concept_space_pre_2: ",concept_space_1_list)

    concept_space_tuple = list(itertools.product(*concept_space_1_list))

    concept_space = []
    for i in concept_space_tuple:
        concept_space.append(list(i))
    #
    # print("concept_space: ",len(concept_space))
    # print("concept_space_glance: ",concept_space[:10])
    # print("single_concept_space: ", len(concept_space[0]))

    sample_list = []
    for i,j in zip(attr_list,label_list):
        tem_tuple = tuple(i) + tuple([j])
        sample_list.append(tem_tuple)
    # print("sample list: ",sample_list)

    index_list = []
    for index1, i in enumerate(concept_space):
        for space_in_space in i:
            for index2,attr in enumerate(attr_list):
                if list(space_in_space[:-1]) == attr:
                    if (space_in_space[-1] != label_list[index2]):
                        index_list.append(index1)

    no_concept_space_index = list(set(index_list))

    version_space_index = []
    for i in range(len(concept_space)):
        if i not in no_concept_space_index:
            version_space_index.append(i)

    version_space = [concept_space[i] for i in version_space_index]

    print(len(version_space_index))
    # print("what is version space:")
    # for i in version_space:
    #     print(i)

    return version_space

def read_std_data(first_occur, data_file = sys.argv[1], encoding='utf-8'):
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

    for index,i in enumerate(test_attr_list):
        for j in range(len(test_attr_list[0])):
            if i[j] == first_occur[j]:
                i[j] = 0
            else:
                i[j] = 1
            # tem_list = []
            # if index == 0:
            #     first_occur.append(i[j])
            #     i[j] = 0
            # else:
            #     if i[j] == first_occur[j]:
            #         i[j] = 0
            #     elif i[j] != first_occur[j]:
            #         i[j] = 1

    return test_attr_list, test_label_list


def take_a_vote(test_attr_list, test_label_list,version_space):
    index_list = []
    vote_list_total = []
    for index1, i in enumerate(test_attr_list):
        i = tuple(i)
        vote_list = []
        for space in version_space:
            for index2,space_in_space in enumerate(space):
                if space_in_space[:-1] == i:
                    vote_list.append(space_in_space[-1])
        vote_list_total.append(vote_list)
    for yes_no_list in vote_list_total:
        Yes_count = 0
        No_count = 0
        for yes_no in yes_no_list:
            if yes_no == 'Yes':
                Yes_count += 1
            else:
                No_count += 1
        print('%s %s'%(Yes_count,No_count))

def main():
    # Input space
    Input_space = 2 * 2 * 2 * 2
    print(Input_space)
    # Concept space
    con_space = 2 ** (Input_space)
    print(con_space)
    attr_list, label_list, first_occur = read_train_data()
    version_space = list_eliminate(attr_list, label_list)
    ##This is for test
    test_attr_list, test_label_list = read_std_data(first_occur)
    take_a_vote(test_attr_list, test_label_list, version_space)

if __name__ == "__main__":
    main()