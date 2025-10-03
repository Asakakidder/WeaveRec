target_domain="sports"
source_domains="['beauty','clothing']"

python traindata/combine.py \
    --target_domain ${target_domain} \
    --source_domains ${source_domains}
