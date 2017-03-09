#! /bin/bash


source ldp-ui.sh

# -----------------------------------------------------------------------------
# Execution

# Executes the selected benchmark
Setup() {

	# Validating logfile folder
	[[ -z ${logfolder} ]] && logfolder="$(pwd)"
	[[ ! -d ${logfolder} ]] && \
		echo "$0 (line $LINENO): logfile folder" ${logfolder}

	logfolder="$(readlink -f ${logfolder})"

	# Creating logfile
	timestamp=$(date "+%m%d%Y%H%M%S")
	logfile="${logfolder}/${timestamp}.log"

	touch ${logfile}
	Info "logfile created" ${logfile}

	# Validating benchmark ID
	[[ ${benchID} -lt 0 || ${benchID} -gt ${#bench[@]} ]] \
		&& Errr "benchmark ID" ${benchID}

}

RunBenchmark() {

	root="$(readlink -m $(pwd)/../${bench[$benchID]})"
	[[ ! -d ${root} ]] && Errr "directory path" ${root}

	bin_seq="${root}/${bench[$benchID]}-seq"
	bin_ldp="${root}/${bench[$benchID]}-ldp"
	bin_par="${root}/${bench[$benchID]}-par"
	bin_lch="${root}/${bench[$benchID]}-lch"

	Info "entering execution folder"
	cd ${logfolder}

	# Creating output files
	errfile_seq="${logfolder}/${timestamp}-seq.err"
	outfile_seq="${logfolder}/${timestamp}-seq.out"

	errfile_ldp="${logfolder}/${timestamp}-ldp.err"
	outfile_ldp="${logfolder}/${timestamp}-ldp.out"

	errfile_par="${logfolder}/${timestamp}-par.err"
	outfile_par="${logfolder}/${timestamp}-par.out"

	errfile_lch="${logfolder}/${timestamp}-lch.err"
	outfile_lch="${logfolder}/${timestamp}-lch.out"

	Info "output and error files created: ${logfolder}/${timestamp}-*"
	touch ${errfile_seq} ${outfile_seq}
	touch ${errfile_ldp} ${outfile_ldp}
	touch ${errfile_par} ${outfile_par}
	touch ${errfile_lch} ${outfile_lch}

	# Setting up inputs
	[[ ${num_runs} -le 0 ]] && Errr "number of runs" ${num_runs}
	Info "running benchmark ${bench[${benchID}]} ${num_runs} times"

	[[ ${num_runs} -le 0 ]] && Errr "input path" ${inputdir}
	Info "running benchmark ${bench[${benchID}]} with tests from ${inputdir}"

	if [[ -z ${infile} ]]; then inputdir="${inputdir}/*.dat"
	else inputdir="${inputdir}/${infile}"; fi
	Info "input files: ${inputdir}"

	# Running benchmark
	for infile in $(ls -v ${inputdir}); do
		[[ ! -f ${infile} ]] && Errr "input file" ${infile}
		${bin_seq} < ${infile} >> ${outfile_seq} 2>> ${errfile_seq}
		${bin_ldp} < ${infile} >> ${outfile_ldp} 2>> ${errfile_ldp}
		${bin_par} < ${infile} >> ${outfile_par} 2>> ${errfile_par}
		${bin_lch} < ${infile} >> ${outfile_lch} 2>> ${errfile_lch}
	done

	for (( i=1; i < ${num_runs}; ++i )); do
		for infile in $(ls -v ${inputdir}); do
			${bin_seq} < ${infile} > /dev/null 2>> ${errfile_seq}
			${bin_ldp} < ${infile} > /dev/null 2>> ${errfile_ldp}
			${bin_par} < ${infile} > /dev/null 2>> ${errfile_par}
			${bin_lch} < ${infile} > /dev/null 2>> ${errfile_lch}
		done
	done

	Info "leaving execution folder"
	cd -

}

# -----------------------------------------------------------------------------
# Main
Arguments $@
Setup
RunBenchmark
