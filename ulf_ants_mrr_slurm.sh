#!/bin/bash

# multi-resolution registration (MRR), using ANTs, to combine AXI+SAG+COR scans into one
# (the script was run on a server via SLURM)
# for details, see: https://www.biorxiv.org/content/10.1101/2024.08.14.607942 (in press at Imaging Neuroscience)
# František Váša, frantisek.vasa@kcl.ac.uk

# data directory
hype_dir=[path_to_folder]/HYPE ### SET PATH TO MAIN FOLDER
in_dir=${hype_dir}/rawdata

# options relevant to multi-resolution registration
mrr_ses=("ses-HFC" "ses-HFE")           # sessions
mrr_acq=("acq-axi" "acq-cor" "acq-sag") # orientations
mrr_con=("T1w" "T2w")                   # contrasts

# script input: current subject ID (used to set subject directory)
sub=$1
sub_dir=${in_dir}/${sub}

# Loop over relevant sessions within each sub_dir
for ses in "${mrr_ses[@]}"; do

    echo "-------"
    echo $ses
    echo "-------"

    # ses directory
    ses_dir="$sub_dir/$ses"
    if [ -d "$ses_dir" ]; then

        # out directory
        out_dir=${hype_dir}/derivatives/ants-mrr/${sub}/${ses}

        # create output directory (if it doesn't exist)
        if [ ! -d ${out_dir} ]; then mkdir -p ${out_dir}; fi

        # loop over contrasts
        for con in "${mrr_con[@]}"; do

            echo "-------"
            echo $con
            echo "-------"

            # check if final output ("mrr") exists
            if [ ! -f ${out_dir}/${sub}_${ses}_${con}_mrr.nii.gz ]; then

                # reference contrast (varies for T1w and T2w data)
                if [ "$con" == "T1w" ]; then
                    ref=ses-GE_acq-MPRAGE_T1w
                elif [ "$con" == "T2w" ]; then
                    ref=ses-GE_T2w
                fi

                # Loop over relevant acquisitions within each sub_dir
                for acq in "${mrr_acq[@]}"; do

                    echo "-------"
                    echo $acq
                    echo "-------"

                    # rigid-register Hyperfine scans to GE (either T1w MPRAGE, or T2w)
                    ants antsRegistrationSyN.sh \
                    -d 3 \
                    -f ${sub_dir}/ses-GE/anat/${sub}_${ref}.nii.gz \
                    -m ${ses_dir}/anat/${sub}_${ses}_${acq}_${con}.nii.gz \
                    -t r \
                    -o ${out_dir}/${sub}_${ses}_${acq}_${con}_ \

                    # rename ANTs outputs to BIDS standard + delete unwanted file(s)
                    mv ${out_dir}/${sub}_${ses}_${acq}_${con}_Warped.nii.gz ${out_dir}/${sub}_${ses}_${acq}_${con}_space-GE.nii.gz
                    rm ${out_dir}/${sub}_${ses}_${acq}_${con}_InverseWarped.nii.gz

                done # for acq in "${mrr_acq[@]}"; do

                # multi-resolution registration for each contrast
                ants antsMultivariateTemplateConstruction2.sh \
                -d 3 -i 12 -c 2 -j 3 -t SyN -m MI -o \
                ${out_dir}/${sub}_${ses}_acq-axi_${con}_space-GE.nii.gz \
                ${out_dir}/${sub}_${ses}_acq-cor_${con}_space-GE.nii.gz \
                ${out_dir}/${sub}_${ses}_acq-sag_${con}_space-GE.nii.gz

                # rename outputs
                mv ${out_dir}/${sub}_${ses}_acq-axi_${con}_space-GE.nii.gztemplate0.nii.gz ${out_dir}/${sub}_${ses}_${con}_mrr.nii.gz

                sleep 60

                # delete extra mrr files
                for item in "$out_dir"/*; do                                                                                                        # loop over items
                    if [ "$item" != "${out_dir}/${sub}_${ses}_T1w_mrr.nii.gz" ] && [ "$item" != "${out_dir}/${sub}_${ses}_T2w_mrr.nii.gz" ]; then   # check if the item is one of the chosen files
                        #echo $item
                        if [[ -d "$item" ]]; then                                                                                                   # check if the item is a directory
                            rm -r "$item"                                                                                                           # delete the directory and its contents recursively
                        else
                            rm "$item"                                                                                                              # delete the file
                        fi
                    fi
                done

            else

                echo "-------"
                echo "mrr output exists!"
                echo "-------"

            fi # check if final output ("mrr") exists

        done # for con in "${mrr_con[@]}"; do 

    fi # if [ -d "$ses_dir" ]; then
done # for ses_dir in "$sub_dir"/ses*; do
