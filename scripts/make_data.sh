for i in Beauty Sports_and_Outdoors Clothing_Shoes_and_Jewelry Grocery_and_Gourmet_Food
do
    python data/raw_process.py --domain ${i}
    python data/data_process.py --domain ${i} --jobnum 93
done
python data/merge.py
cp -r data/processed/sports/prompt/test.jsonl testdata/sports.jsonl
cp -r data/processed/beauty/prompt/test.jsonl testdata/beauty.jsonl
cp -r data/processed/clothing/prompt/test.jsonl testdata/clothing.jsonl
cp -r data/processed/food/prompt/test.jsonl testdata/food.jsonl