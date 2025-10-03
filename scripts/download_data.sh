for name in Beauty Sports_and_Outdoors Clothing_Shoes_and_Jewelry Grocery_and_Gourmet_Food
do
    wget -P data/rawdata https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_${name}_5.json.gz
    wget -P data/rawdata https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_${name}.json.gz
done