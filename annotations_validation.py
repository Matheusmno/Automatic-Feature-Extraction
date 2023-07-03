import re

def check_annotations(file: dict) -> bool:
    match_pattern = r"[cspt]_\w+_(start|stop)"
    annotations = file["header"]["annotations"]
    
    if any(map(lambda x : not re.match(match_pattern, x[-1]), annotations)):
        pattern_mismatches = [i for i, val in enumerate(map(lambda x : re.match(match_pattern, x[-1]), annotations)) if val is None]
        print(f"There are {len(pattern_mismatches)} annotations out of the pattern.")
        #for idx in pattern_mismatches:
        #    print(idx, file["header"]["annotations"][idx])
    
    if not annotations:
        print("There are no annotations in the file.")
        return False
    
    elif check_C_annotations(list(filter(lambda x : re.match(r"c_\w+_\w+", x[-1]), annotations))):
        if check_T_annotations(list(filter(lambda x : re.match(r"[spt]_", x[-1]), annotations))):
            return True
    return False

def ann_time_to_string(time_seconds):

    time = time_seconds
    h = int(time / 60 / 60)
    m = int((time - h * 60 * 60) / 60)
    s = int(time - h * 60 * 60 - m * 60)
    string = str(h) + ":" + str(m) + ":" + str(s)
    return string

def check_T_annotations(T_ann_list):    
    check = True
    if T_ann_list:
        # check test annotations
        for i in range(0, len(T_ann_list)):

            # check annotation format
            split_string = T_ann_list[i][2].split("_")
            if len(split_string) != 3:
                print("Wrong Probe Annotation Format " + T_ann_list[i][2] + "t_<label>_start/stop required")

            # check if stop annotation is available for start annotation
            if "start" in T_ann_list[i][2]:
                ann_start = T_ann_list[i][2]
                found = False
                for j in range(i+1, len(T_ann_list)):
                    if ann_start[0:-6] == T_ann_list[j][2][0:-5]:
                        found = True
                        break

                if not found:
                    check = False
                    time = T_ann_list[i][0]
                    time_string = ann_time_to_string(time)
                    print("No Stop For Test " + ann_start + " " + time_string)

            # check if start annotation is available for stop annotation
            if "stop" in T_ann_list[i][2]:
                ann_stop = T_ann_list[i][2]
                found = False
                j = i-1
                while j >= 0:
                    if ann_stop[0:-5] == T_ann_list[j][2][0:-6]:
                        found = True
                        break
                    j = j - 1

                if found == False:
                    check = False
                    time = T_ann_list[i][0]
                    time_string = ann_time_to_string(time)
                    print("No Start For Test " + ann_stop + " " + time_string)

    return check

def check_C_annotations(C_ann_list):

    # check category annotations
    i = 0
    check = True
    if C_ann_list:
        while i < len(C_ann_list)-1:

            # check annotation format
            split_string = C_ann_list[i][2].split("_")
            if len(split_string) != 3:
                print("Wrong Category Annotation Format " + C_ann_list[i][2] + "c_<label>_start/stop required")

            if "start" in C_ann_list[i][2] and "stop" in C_ann_list[i+1][2]:
                ann1 = C_ann_list[i][2]
                ann2 = C_ann_list[i+1][2]

                # check identical category label
                if ann1.split("_")[1] != ann2.split("_")[1]:
                    time_string_1 = ann_time_to_string(C_ann_list[i][0])
                    time_string_2 = ann_time_to_string(C_ann_list[i+1][0])

                    print("Category Mismatch for " + C_ann_list[i][2] + " at " + time_string_1 +
                        " and " + C_ann_list[i+1][2] + " at " + time_string_2)
                    check = False

                i = i + 2

            elif "start" in C_ann_list[i][2] and "stop" not in C_ann_list[i + 1][2]:
                time_string = ann_time_to_string(C_ann_list[i][0])
                print("Missing Stop Annotations for " + C_ann_list[i][2] + " at " + time_string)
                check = False
                i = i + 1

    return check