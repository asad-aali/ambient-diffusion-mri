# Calculate Reference Scores
python fid.py ref --data=/csiNAS/sidharth/updated-edm/ref_data --dest=fid-refs/ref-dataset.npz

# Calculate FID
torchrun --standalone --nproc_per_node=1 fid.py calc --images=/csiNAS/sidharth/updated-edm/ambient_priors/ncsnv_priors --ref=fid-refs/ref-dataset.npz