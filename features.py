# for test, skip glove to speed up
TEST = False

# add or remove any features
add_title = True
add_body = True

add_file_list_sim = True
add_overlap_files_len = True
add_code_sim1 = True
add_code_sim2 = True
add_location_sim1 = True
add_location_sim2 = True
add_pattern = True
add_time = True

fileName = "remove"

"--------------------------------"
if not add_title:
    fileName = fileName + "_title"
if not add_body:
    fileName = fileName + "_body"
if not add_file_list_sim:
    fileName = fileName + "_filesim"
if not add_overlap_files_len:
    fileName = fileName + "_filelen"
if not add_code_sim1:
    fileName = fileName + "_code1"
if not add_code_sim2:
    fileName = fileName + "_code2"
if not add_location_sim1:
    fileName = fileName + "_loca1"
if not add_location_sim2:
    fileName = fileName + "_loca2"
if not add_pattern:
    fileName = fileName + "_pattern"
if not add_time:
    fileName = fileName + "_time"
"--------------------------------"