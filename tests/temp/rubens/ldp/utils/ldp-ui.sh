#! /bin/bash


# Global constant variables
declare -ar bench=( $(cat bench.cfg) )

# Global variables
declare -i benchID=-1
declare -i num_runs=-1
declare -i display_width=80

declare infile logfolder inputdir
declare timestamp outfile errfile logfile

# -----------------------------------------------------------------------------
# User Interface
Arguments() {

	[[ $# -eq 0 ]] && PrintHelp

	local -i i=1
	while [[ $i -le $# ]]; do

		arg=${@:$i:1}

		case $arg in
		-h )
			PrintHelp
			;;
		--help )
			(( ++i ))
			display_width=${@:$1:1}
			PrintHelp
			;;
		-b )
			(( ++i ))
			benchID=${@:$i:1}
			;;
		-i )
			(( ++i ))
			infile="${@:$i:1}"
			;;
		-I )
			(( ++i ))
			inputdir=${@:$i:1}
			;;
		--list-bench )
			ListBenchmarks
			;;
		-l )
			(( ++i ))
			timestamp=${@:$i:1}
			;;
		-L )
			(( ++i ))
			logfolder=${@:$i:1}
			;;
		-n )
			(( ++i ))
			num_runs=${@:$i:1}
			;;
		* )
			echo "$0 (line $LINENO): unknown option #$i '$arg'"
			;;
		esac

		(( ++i ))

	done

}

# Help
PrintHelp() {
	(echo .ll ${display_width}; cat arguments.1) | nroff -man | sed 's/.//g'
	exit 1
}

# Lists available benchmarks
ListBenchmarks() {
	(echo .ll ${display_width}; cat benchmarks.1) | nroff -man | sed 's/.//g'
	exit 1
}

# Dumps execution messages
Debug() {
	[[ ! -f ${logfile} ]] && Error "invalid logfile!"
	echo "[$(date '+%m-%d-%Y %H:%M:%S')] -- $1" >> ${logfile}
}

Info() {
	Debug "Info: $1"
}

Warn() {
	Debug "Warn: $1"
}

Errr() {

	if [[ $# -eq 1 ]]; then Debug "Error: invalid $1, line $(caller)"
	elif [[ $# -eq 2 ]]; then Debug "Error: invalid $1 ($2), line $(caller)"
	else Debug "Error: unknow error, line $(caller)"
	fi

	exit 1

}

Error() {
	echo "$(caller): $1"
	exit 1
}

