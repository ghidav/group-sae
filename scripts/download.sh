git lfs install

git clone https://huggingface.co/Group-SAE/pythia_160m-topk
git clone https://huggingface.co/Group-SAE/pythia_410m-topk
git clone https://huggingface.co/Group-SAE/pythia_1b-topk



mkdir saes
mv pythia_160m-topk saes/pythia_160m-topk
cp saes/pythia_160m-topk/cluster/config.json saes/pythia_160m-topk/baseline
mv pythia_410m-topk saes/pythia_410m-topk
cp saes/pythia_410m-topk/cluster/config.json saes/pythia_410m-topk/baseline
mv pythia_1b-topk saes/pythia_1b-topk
cp saes/pythia_1b-topk/cluster/config.json saes/pythia_1b-topk/baseline
