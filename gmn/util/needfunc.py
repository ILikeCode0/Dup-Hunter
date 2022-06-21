import copy
import re
from git import *



def parse_diff(file_name, diff):
    parts = re.split('(@@.*?-.*?\+.*?@@)', diff)
    start_with_plus_regex = re.compile('^\++')
    start_with_minus_regex = re.compile('^\-+')

    add_diff_code = ""
    del_diff_code = ""
    add_code_line = 0
    del_code_line = 0
    add_location_set = []
    del_location_set = []

    parts_len = len(parts)
    for i in range(1, parts_len, 2):
        location = parts[i]
        part = parts[i + 1]

        try:
            add, dele = location.replace('@', '').strip().split(' ')
        except:
            continue

        if len(part) >= 100 * 1024:
            continue

        try:
            if ('-' in add) and ('+' in dele):
                add, dele = dele, add

            if ',' in add:
                add_location, add_line = add[1:].split(',')
            else:
                add_location = 0
                add_line = add[1:]

            if ',' in dele:
                del_location, del_line = dele[1:].split(',')
            else:
                del_location = 0
                del_line = dele[1:]

            lines_of_code = [x.strip() for x in part.splitlines()]

            added_lines_of_code = filter(lambda x: (x) and (x[0] == '+'), lines_of_code)
            added_lines_of_code = [start_with_plus_regex.sub('', x) for x in added_lines_of_code]

            deleted_lines_of_code = filter(lambda x: (x) and (x[0] == '-'), lines_of_code)
            deleted_lines_of_code = [start_with_minus_regex.sub('', x) for x in deleted_lines_of_code]

            add_diff_code += '\n'.join(added_lines_of_code) + '\n'
            del_diff_code += '\n'.join(deleted_lines_of_code) + '\n'

            add_code_line += len(added_lines_of_code)
            del_code_line += len(deleted_lines_of_code)

            add_location_set.append([int(add_location), int(add_line)])
            del_location_set.append([int(del_location), int(del_line)])
        except Exception as e:
            print('Parse Error:', e)

    return {"name": file_name,
            "LOC": {
                "add": add_code_line,
                "del": del_code_line,
            },
            "location": {
                "add": add_location_set,
                "del": del_location_set,
            },
            "add_code": add_diff_code,
            "del_code": del_diff_code,
            }


def get_file_list(list):
    return [x["name"] for x in list]


def check_pattern(A, B):
    ab_num = set([A["number"], B["number"]])
    a_text = str(A["title"]) + ' ' + str(A["body"])
    b_text = str(B["title"]) + ' ' + str(B["body"])

    a_set = set(get_numbers(a_text) + get_version_numbers(a_text)) - ab_num
    b_set = set(get_numbers(b_text) + get_version_numbers(b_text)) - ab_num
    if a_set & b_set:
        return 1
    else:
        def get_reasonable_numbers(x):
            return get_pr_and_issue_numbers(x) + get_version_numbers(x)

        a_set = set(get_reasonable_numbers(a_text)) - ab_num
        b_set = set(get_reasonable_numbers(b_text)) - ab_num
        if a_set and b_set and (a_set != b_set):
            return -1
        return 0


def get_pull_on_overlap(pull, overlap_set):
    new_pull = copy.deepcopy(pull)
    new_pull["file_list"] = list(filter(lambda x: x["name"] in overlap_set, new_pull["file_list"]))
    return new_pull

