clear;
rm log.txt
rm nohup.txt
# rm -rf output
ROOT=savecode/
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir snapshot
# CUDA_VISIBLE_DEVICES=0,1 python -u $ROOT/main.py --log log.txt --config config.json
# nohup python $ROOT/main.py --log log.txt --config config.json > nohup.txt &
python $ROOT/main.py --log log.txt --config config.json

