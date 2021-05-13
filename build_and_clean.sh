python setup.py build
ext_file=`ls build/ | grep lib`
for file in $ext_file
do
  cp 'build/'$file'/*' .
done
rm -rf 'build'
cd ..