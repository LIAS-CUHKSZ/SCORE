Tune the label_remapping.txt to delete unwanted labels and merge similar labels for each dataset.
label_remapping.txt is a txt file with two columns, the first column is the original label, and the second column is the new label
1. if the second column is zero, the label is deleted
2. if the second column is not zero, the original label is remapped to the new label 

You can refer to dictionary.txt for the semantics.
dictionary.txt automatically changes according to label_remapping.txt, after running construct_raw_line_map.py.