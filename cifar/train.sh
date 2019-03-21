# wocao
CFG_PATH=./cfgs/accord_std/

FILES=$(ls $CFG_PATH)
echo $FILES
for i in $FILES; do
    echo $i
    python train.py --cfg=$CFG_PATH$i
done