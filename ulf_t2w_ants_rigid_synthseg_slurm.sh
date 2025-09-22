#!/bin/bash

# rigid registration of HFC+HFE (64mT) synthseg outputs to GE (3T) within individuals, for quantification of dice overlap
# (the script was run on a server via SLURM)
# for details, see: https://www.biorxiv.org/content/10.1101/2024.08.14.607942 (in press at Imaging Neuroscience)
# František Váša, frantisek.vasa@kcl.ac.uk

# data directory
hype_dir=[path_to_folder]/HYPE ### SET PATH TO MAIN FOLDER
in_dir=${hype_dir}/derivatives/synthseg_t2w

# script input: current subject ID (used to set subject directory)
sub=$1
sub_dir=${in_dir}/${sub}

# output directory
out_dir=${hype_dir}/derivatives/synthseg_t2w-rigid/${sub}

# create output directory (if it doesn't exist)
if [ ! -d ${out_dir} ]; then mkdir -p ${out_dir}; fi

# concatenate all synthseg outputs (ran separately)
in_t2_nii=${out_dir}/${sub}_synthseg_t2w_in_all_nii.txt		# t2w inputs
out_seg_nii=${out_dir}/${sub}_synthseg_t2w_out_all_nii.txt 	# synthseg outputs
if [ ! -f ${in_t2_nii} ]; then cat ${sub_dir}/${sub}_synthseg_t2w_in_nii.txt ${sub_dir}/${sub}_synthseg_t2w_mrr_in_nii.txt > ${in_t2_nii}; fi 		# t2w inputs
if [ ! -f ${out_seg_nii} ]; then cat ${sub_dir}/${sub}_synthseg_t2w_out_nii.txt ${sub_dir}/${sub}_synthseg_t2w_mrr_out_nii.txt > ${out_seg_nii}; fi # synthseg outputs

# reference t2: 
#ref_t2_nii=$(head -n 2 ${in_t2_nii} | tail -1 | cut -d ':' -f 1)
ref_t2_nii=$(awk '/ses-GE_T2w/' ${in_t2_nii})

echo "-------"
echo "-------"
echo "reference: $ref_t2_nii"
echo "-------"
echo "-------"

### Loop over text files
while IFS= read -r t2w_path && IFS= read -r seg_path <&3; do

	# test
	#echo "in t2w nii: $t2w_path"
	#echo "out seg nii: $seg_path"

	t2w=$(basename -s .nii.gz ${t2w_path}) # file

	# do not register the reference...
	if [[ ${t2w_path} == ${ref_t2_nii} ]]; then

		continue

	# ...but register all other scans
	else

		if  [ ! -f ${out_dir}/${t2w}_synthseg_to_T2w.nii.gz ]; then

			# delete existing ${t2w} files
			rm ${out_dir}/${t2w}_*

			echo "-------"
			echo "estimating transform: $t2w_path"
			echo "-------"

		    # rigid-register each scan to GE T2w
		    ants antsRegistrationSyN.sh \
		    -d 3 \
		    -f ${ref_t2_nii} \
		    -m ${t2w_path} \
		    -t r \
		    -o ${out_dir}/${t2w}_

		    # rename ANTs outputs to BIDS standard + delete unwanted file(s)
		    mv ${out_dir}/${t2w}_Warped.nii.gz ${out_dir}/${t2w}_space-T2w.nii.gz
		    rm ${out_dir}/${t2w}_InverseWarped.nii.gz

			echo "-------"
			echo "applying transform: $seg_path"
			echo "-------"

		   	# apply registration to segmentation file
			ants antsApplyTransforms \
			-n NearestNeighbor \
			-e 0 \
			-d 3 \
			-i ${seg_path} \
			-r ${ref_t2_nii} \
			-o ${out_dir}/${t2w}_synthseg_to_T2w.nii.gz \
			-t ${out_dir}/${t2w}_0GenericAffine.mat

		fi

	fi # if [[ ${t2w_path} != ${ref_t2_nii} ]]; then

done < ${in_t2_nii} 3< ${out_seg_nii}
