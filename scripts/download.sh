git lfs install
git clone git@hf.co:Group-SAE/pythia_160m-jr
git clone git@hf.co:Group-SAE/pythia_160m-topk

git clone git@hf.co:Group-SAE/pythia_410m-jr
git clone git@hf.co:Group-SAE/pythia_410m-topk

#git clone git@hf.co:Group-SAE/gemma2_2b-jr
#git clone git@hf.co:Group-SAE/gemma2_2b-topk


mkdir saes
mv pythia_160m-jr saes/pythia_160m-jr
mv pythia_160m-topk saes/pythia_160m-topk

mv pythia_410m-jr saes/pythia_410m-jr
mv pythia_410m-topk saes/pythia_410m-topk

#mv gemma2_2b-jr saes/gemma2_2b-jr
#mv gemma2_2b-topk saes/gemma2_2b-topk
