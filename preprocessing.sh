#!/usr/bin/env bash

pandoc_cmd="pandoc"

Usage() {
    cat <<- EOT

  Usage: ${0##/*/} [options] -i odt -o md

  Options:
  -i|input    Directory to ODT files
  -o|output   Directory to MD file
  -d|debug    Debug Mode
  -h|help     Display this message

EOT
}

checkDIR() {
    echo -n "[+] Checking the directory <$1> ......"
    if [ ! -d  $1 ]
    then
        echo "[*Failed*]"
        exit 1
    else
        echo "[OK]"
    fi
}

odt2md() {
    input_file=$1
    input_name=$(basename $1)
    outdir=$2
    out_name="${input_name%%.odt}.md"
    cmd_args="-f odt -t markdown -o $outdir/$out_name"

    if [ "$debug" = true ]
    then
        $pandoc_cmd $cmd_args $input_file	 
    else
        $pandoc_cmd $cmd_args $input_file >/dev/null 2>&1
    fi
 
    if [ $? -eq 0 ]
    then
        return 0
    else
        return 1
    fi
}

revisions() {
    local mdfile
    mdfile=$1
    # Search: **修正日期：**民國 112 年 08 月 16 日
    # Replace: 修正日期：民國 112 年 08 月 16 日
    sed -i "s/^\*\*修正日期：\*\*/修正日期：/g" $mdfile

    # Remmove the line with the '\'
    sed -i '/^\\$/d' $mdfile
}

# Serch: **法規名稱：**性別平等工作法
# Replace: ### 法規名稱：性別平等工作法
head_h3() {
    local mdfile
    mdfile=$1
    sed -i 's/^\*\*\(法規名稱：\)\*\*\(.*\)$/### \1\2/' $mdfile
}

# Search: ** 第 一 章 總則**
# Search: **第 一 編 總則**
# Replace: #### 第 一 章 總則
# Replace: #### 第 一 編 總則 
head_h4() {
    local mdfile
    mdfile=$1
    sed -i 's/^\*.*\(第\ .*\ 章 .*\)\*\*.*$/#### \1/' $mdfile
    sed -i 's/^\*.*\(第\ .*\ 編 .*\)\*\*.*$/#### \1/' $mdfile
}

# Search: ** 第 一 節 通則**
# Search: ** 第 三 款 無因管理**
# Replace: ##### 第 一 節 通則 
# Replace: ##### 第 三 款 無因管理
head_h5() {
    local mdfile
    mdfile=$1
    sed -i 's/^\*.*\(第\ .*\ 節 .*\)\*\*.*$/##### \1/' $mdfile
    sed -i 's/^\*.*\(第\ .*\ 款 .*\)\*\*.*$/##### \1/' $mdfile
}

# Search: **第 1 條**
# Replace: ###### 第 1 條
head_h6() {
    local mdfile
    mdfile=$1
    sed -i 's/^\*\*\(第\ .*\ 條\).*$/###### \1/' $mdfile
}

# To verify the fix, run the following command.
# Return nothing that means the fix was applied successfully.
# #> grep "^[123456789]$" your.md
fix_ordered_list() {
    local mdfile
    local n
    mdfile=$1
    for ((i=1;i<=10;i++))  # The loop for times is required for filtering the whole text.
    do
        sed -i 'N;s/^\([123456789]\)\n/\1 /' $mdfile
    done
}

########## Main Program #############
if [ $# -lt 4 ]
then
    # invalid arguments
    Usage
    exit 1
fi

while getopts "i:o:dh" opt
do
    case $opt in
      i) odtdir=$OPTARG;;
      o) mddir=$OPTARG;;
      d) debug=true;;
      h) Usage; exit ;;
      \? )  echo -e "\n  Option does not exist : $OPTARG\n"
         Usage; exit 1   ;;
    esac
done
shift $(($OPTIND-1))

if ! which $pandoc_cmd >/dev/null 2>&1
then
    echo "[!] Pandoc is required for this program, please install it first."
    echo
    echo "Recommend: Following the command below to install Pandoc."
    echo "With Ubuntu/Debian:"
    echo "  apt update && apt install pandoc"
    echo "With RedHat:"
    echo "  yum install pandoc"
    echo
    exit 1
fi

echo
checkDIR $odtdir
checkDIR $mddir
workdir="$( cd $( dirname "$0" ) && pwd )"
cd $workdir
input_count=$(ls -l $odtdir/*.odt 2>/dev/null | grep -v "^d" | grep -v "^total" | wc -l)
if [ $input_count -gt 0 ]
then
    echo "[+] Founded ($input_count) ODT files."
else
    echo "[!] Not found any ODT files, the process is being terminated"
    exit 1
fi

echo "[+] Processing the files."
out_count=0
for odtf in $odtdir/*.odt
do
    odtf_name=$(basename $odtf)
    name=${odtf_name%%.odt}
    if odt2md "$odtf" "$mddir"
    then
        mdf_name="$name.md"
        revisions "$mddir/$mdf_name"
        head_h3 "$mddir/$mdf_name"
        head_h4 "$mddir/$mdf_name"
        head_h5 "$mddir/$mdf_name"
        head_h6 "$mddir/$mdf_name"
        fix_ordered_list "$mddir/$mdf_name"
	echo -n "--> [$mdf_name] "
	(( out_count++ ))
    else
	echo -n "--> **[$mdf_name]** "
    fi
done
echo
echo "[+] Done: "
echo " - Processed ($input_count) files."
echo " - Converted ($out_count) files."
