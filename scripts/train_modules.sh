target_domain="sports"
source_domains="beauty clothing"
# Note that source_domains here is a string with each source domain separated by a space.

sh scripts/train_lora.sh ${target_domain}
for domain in $source_domains
do
    hybrid_domain="${target_domain}_${domain}"
    sh scripts/train_lora.sh ${hybrid_domain}
    echo "====================================="
    echo "Training Done for ${hybrid_domain}..."
    echo "====================================="
done