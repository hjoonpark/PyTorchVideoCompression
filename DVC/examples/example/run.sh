clear;
rm log.txt
rm nohup.txt
ROOT=savecode/
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir snapshot
# nohup python $ROOT/main.py --log log.txt --config config.json > nohup.txt &
# CUDA_VISIBLE_DEVICES=0,1 python -u $ROOT/main.py --log log.txt --config config.json
python $ROOT/main.py --log log.txt --config config.json

