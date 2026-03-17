nvidia-smi
ls -lh cardata.zip
pip install --upgrade gdown
gdown --folder 1R0e-xgdLF9aaw7mpCe436kvonK8cwymx
pip install gdown
gdown --id 1Ybb0jgwwym-v7AEeFepTlY-pur_CN9ft
gdown --id 1psn16SX8xlZPOCHTdTeVjStXsFPCAAyX
mkdir -p Video_Tensors/positive
mkdir -p Video_Tensors/negative
unzip positive.zip -d Video_Tensors/positive
unzip negative.zip -d Video_Tensors/negative
mv positive/positive/* positive/
ls -R | grep ":$" | head -n 10
cd Video_Tensors
cd positive
mv positive/* .
rmdir positive
pip install torch torchvision scikit-learn numpy pillow
nvidia-smi
python main.py
python cnn.py
python cnn.py
torch.cuda.empty_cache() python cnn.py
python cnn.py
python cnn.py
python resnet.py
torch.cuda.empty_cache() python resnet.py
python resnet.py
AttributeError: 'CustomResidualExtractor' object has no attribute 'fc'
torch.cuda.empty_cache() python resnet.py
python resnet.py
python resnet.py
python resnet.py
python resnet.py
pkill -9 python
python resnet.py
python resnet.py
python resnet.py
pkill -9 python
python resnet.py
python resnet.py
cd /teamspace/studios/this_studio
nano .gitignore
cat .gitignore
git init
