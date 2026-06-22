source /home/hongjiacheng/Codes/gs_lab/WorldAct_Codex/openai.env

export CUDA_VISIBLE_DEVICES=7
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11 
export CUDAHOSTCXX=/usr/bin/g++-11 

python scripts/render_chunks.py \
    --ply 0001_839920.ply \
    --output-dir outputs/chunk_preview \
    --device cuda \
    --num-views 4 \
    --chunk-target-gaussians 120000 \
    --chunk-overlap 0.8 \
    --max-chunks 16 \
    --camera-width 512 \
    --camera-height 512 \
    --save-ply