#! /bin/bash


benchid=(
	"merge-ldp-__"
	"merge-ldp-bi"
	"merge-lch-__"
	"merge-lch-bi"
	"merge-seq-__"
	"quick-ldp-__"
	"quick-ldp-bi"
	"quick-lch-__"
	"quick-lch-bi"
	"quick-seq-__"
)

for i in ${benchid[@]}; do
	echo -ne "$i\t"
	grep -oP "$i \K\d+\.\d+" times.dat | tr '\n' ', ' | sed 's/,/, /g'
	echo
done
