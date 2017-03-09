#! /bin/bash


case $1 in
--bin )
	/dev/urandom | tr -d \''\\'\` | head -c ${2}
	;;
--text )
	tr -dc '[:alnum:] ,.!?;' < /dev/urandom | tr -d \''\\'\` | head -c ${2}
	;;
--rand )
	tr -dc '[:alnum:] ,.!?;' < /dev/urandom | tr -d \''\\'\` | head -c ${2}
	;;
* )
	echo "$0 [--bin, --text, --rand] size"
	;;
esac
