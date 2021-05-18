ext_file=`ls build/ | grep lib`
ext_file=$ext_file/`ls build/$ext_file`
cp build/$ext_file .
rm -rf build