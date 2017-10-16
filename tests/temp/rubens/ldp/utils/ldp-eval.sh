#! /bin/bash

source ldp-ui.sh


declare seq ldp par

# Formats numbers from the results obtained
FormatNumbers() {

	seq="${logfolder}/${timestamp}-seq.res"
	ldp="${logfolder}/${timestamp}-ldp.res"
	par="${logfolder}/${timestamp}-par.res"

	local p="\d+\.\d+(?= million)"
	errfile_seq="${logfolder}/${timestamp}-seq.err"
	errfile_ldp="${logfolder}/${timestamp}-ldp.err"
	errfile_par="${logfolder}/${timestamp}-par.err"

	local num_files=$(ls -1v ${inputdir} | wc -l)
	local s=$(printf "N;%.0s" $(seq 2 ${num_files}))
	[[ ${num_files} -le 1 ]] && s="p" || s="${s} s/\n/\t/g; p"

	sequence="$(echo {256..2048..256})"
	# echo "256 512 1024 2048 4096" > ${tmp}
	# echo "4 8 12 16 20 24 28 32 36" > ${tmp}

	tmp=$(mktemp)
	echo "${sequence}" > ${tmp}
	grep -oP "${p}" ${errfile_seq} | sed -n "${s}" >> ${tmp}
	column -t ${tmp} > ${seq}

	echo "${sequence}" > ${tmp}
	grep -oP "${p}" ${errfile_ldp} | sed -n "${s}" >> ${tmp}
	column -t ${tmp} > ${ldp}

	echo "${sequence}" > ${tmp}
	grep -oP "${p}" ${errfile_par} | sed -n "${s}" >> ${tmp}
	column -t ${tmp} > ${par}

}


# Main
Arguments $@
FormatNumbers
# ldp.R ${ldp} ${par} ${seq} "${logfolder}/${}"
