#!/bin/bash

# synthseg (robust, with cortical parcellation)
# (the script was run on a server via SLURM)
# for details, see: https://www.biorxiv.org/content/10.1101/2024.08.14.607942 (in press at Imaging Neuroscience)
# František Váša, frantisek.vasa@kcl.ac.uk

# data directory
hype_dir=[path_to_folder]/HYPE ### SET PATH TO MAIN FOLDER
in_dir=${hype_dir}/rawdata

# options relevant to synthseg
ss_ses=("ses-GE" "ses-HFC" "ses-HFE")   # sessions
ss_mod=("anat")                         # modalities

# script input: current subject ID (used to set subject directory)
sub=$1
sub_dir=${in_dir}/${sub}

# output directory
out_dir=${hype_dir}/derivatives/synthseg_t2w/${sub}

# create output directory (if it doesn't exist)
if [ ! -d ${out_dir} ]; then mkdir -p ${out_dir}; fi

# files with paths to inputs and outputs
in_nii=${out_dir}/${sub}_synthseg_t2w_in_nii.txt
out_nii=${out_dir}/${sub}_synthseg_t2w_out_nii.txt
out_vol=${out_dir}/${sub}_synthseg_t2w_out_vol.txt

### generate input text files
# loop over sessions within each sub_dir
for ses in "${ss_ses[@]}"; do

    echo "-------"
    echo $ses
    echo "-------"

    # ses directory
    ses_dir="$sub_dir/$ses"
    if [ -d "$ses_dir" ]; then

        # loop over modalities
        for mod in "${ss_mod[@]}"; do

            # mod directory
            mod_dir="$ses_dir/$mod"
            if [ -d ${mod_dir} ]; then # check if modality exists (*neuromix only exists for ses-GE*)

                echo "-------"
                echo $mod
                echo "-------"

                # write *T1.nii.gz files within each modality to the text file
                for t2w_path in ${mod_dir}/*T2w.nii.gz; do

                    t2w=$(basename -s .nii.gz ${t2w_path}) # file

                    echo ${t2w_path} >> ${in_nii}                         # in_nii
                    echo ${out_dir}/${t2w}_synthseg.nii.gz >> ${out_nii}  # out_nii
                    echo ${out_dir}/${t2w}_synthseg.csv >> ${out_vol}     # out_vol

                done # for t2w in ${mod_dir}/*; do

            fi

        done # for mod in "${ss_mod[@]}"; do

    fi # if [ -d "$ses_dir" ]; then
done # for ses_dir in "$sub_dir"/ses*; do

# run synthseg
mri_synthseg \
--i ${in_nii} \
--o ${out_nii} \
--parc \
--robust \
--vol ${out_vol}
